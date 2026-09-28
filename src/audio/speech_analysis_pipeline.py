"""Privacy-safe local speech analysis built around a single microphone stream."""

from __future__ import annotations

from dataclasses import dataclass, replace
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
import time

import numpy as np

from src.audio.audio_capture import AudioCaptureService
from src.audio.audio_event_detector import AudioState, DEFAULT_MODEL_PATH, YamNetAudioDetector
from src.audio.loudness_analyzer import LoudnessAnalyzer, LoudnessState
from src.audio.speech_transcriber import FasterWhisperTranscriber, TranscriptResult
from src.audio.text_safety_classifier import BilingualSafetyClassifier, SafetyResult
from src.audio.transcript_store import TranscriptStore, export_readable_transcript


TRANSCRIPT_TTL_MS = 15_000


@dataclass(frozen=True)
class SpeechAnalysisState:
    audio_available: bool = False
    audio_top_label: str = "Unavailable"
    audio_top_confidence: float = 0.0
    raised_voice_active: bool = False
    horn_active: bool = False
    siren_active: bool = False
    music_active: bool = False
    quiet_active: bool = False
    speech_active: bool = False
    language: str = "unknown"
    language_probability: float = 0.0
    transient_transcript: str = ""
    transcript_visible: bool = False
    transcript_display_enabled: bool = True
    transcript_recording_enabled: bool = False
    recorded_transcript_count: int = 0
    dropped_utterance_count: int = 0
    last_transcript_saved_ms: int = 0
    transcription_pending: bool = False
    transcription_status: str = "Not listening"
    voice_level: str = "QUIET"
    loudness_dbfs: float = -120.0
    loud_voice_active: bool = False
    safety_flagged: bool = False
    safety_categories: tuple[str, ...] = ()
    safety_confidence: float = 0.0
    safety_model_available: bool = False
    timestamp_ms: int = 0
    transcript_timestamp_ms: int = 0
    error: str = "Audio disabled"
    analysis_error: str = ""

    @classmethod
    def unavailable(cls, error: str) -> "SpeechAnalysisState":
        return cls(error=error)


class EnergyUtteranceGate:
    """Small scheduler gate that retains context before a confirmed utterance."""

    def __init__(
        self,
        *,
        active_dbfs: float | None = None,
        minimum_active_dbfs: float = -65.0,
        maximum_active_dbfs: float = -30.0,
        noise_margin_db: float = 12.0,
        start_chunks: int = 2,
        pre_roll_chunks: int = 6,
        end_chunks: int = 20,
        max_samples: int = 320_000,
    ):
        if start_chunks < 1:
            raise ValueError("start_chunks must be at least one")
        if pre_roll_chunks < start_chunks:
            raise ValueError("pre_roll_chunks cannot be smaller than start_chunks")
        self.active_dbfs = active_dbfs
        self.minimum_active_dbfs = minimum_active_dbfs
        self.maximum_active_dbfs = maximum_active_dbfs
        self.noise_margin_db = noise_margin_db
        self.start_chunks = start_chunks
        # Keep extra in-memory lead-in without delaying the start decision.
        # Quiet Arabic consonants such as the opening ح in حضرتك can occur
        # before two consecutive chunks cross the energy threshold.
        self.pre_roll_chunks = pre_roll_chunks
        self.end_chunks = end_chunks
        self.max_samples = max_samples
        self._noise_floor_dbfs = -80.0
        self._current_threshold_dbfs = active_dbfs if active_dbfs is not None else -60.0
        self.last_chunk_speech = False
        self._active = False
        self._above = 0
        self._below = 0
        self._pre_roll: list[np.ndarray] = []
        self._utterance: list[np.ndarray] = []

    def update(self, samples: np.ndarray, dbfs: float) -> tuple[bool, np.ndarray | None]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if self.active_dbfs is None and not self._active and dbfs < self._current_threshold_dbfs:
            self._noise_floor_dbfs = 0.92 * self._noise_floor_dbfs + 0.08 * dbfs
            adaptive_threshold = self._noise_floor_dbfs + self.noise_margin_db
            self._current_threshold_dbfs = min(
                self.maximum_active_dbfs,
                max(self.minimum_active_dbfs, adaptive_threshold),
            )
        voice_energy = dbfs >= self._current_threshold_dbfs
        self.last_chunk_speech = voice_energy
        self._pre_roll.append(samples)
        self._pre_roll = self._pre_roll[-self.pre_roll_chunks :]

        if not self._active:
            self._above = self._above + 1 if voice_energy else 0
            if self._above >= self.start_chunks:
                self._active = True
                self._below = 0
                self._utterance = list(self._pre_roll)
            return self._active, None

        self._utterance.append(samples)
        self._below = self._below + 1 if not voice_energy else 0
        sample_count = sum(len(part) for part in self._utterance)
        if self._below < self.end_chunks and sample_count < self.max_samples:
            return True, None

        result = np.concatenate(self._utterance)
        self._active = False
        self._above = 0
        self._below = 0
        self._utterance = []
        return False, result

    @property
    def current_threshold_dbfs(self) -> float:
        return self._current_threshold_dbfs

    def flush(self) -> np.ndarray | None:
        """Return the current segment when the caller deliberately ends listening."""
        if not self._active or not self._utterance:
            return None
        result = np.concatenate(self._utterance)
        self._active = False
        self._above = 0
        self._below = 0
        self._utterance = []
        return result


class SpeechAnalysisPipeline:
    """Coordinates YAMNet, loudness, Whisper, and text safety without saving audio."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        yamnet_model_path=DEFAULT_MODEL_PATH,
        whisper_model: str = "large-v3-turbo",
        transcription_device: str = "auto",
        transcription_compute_type: str = "auto",
        transcription_language: str | None = None,
        display_transcript: bool = True,
        mask_unsafe_transcript: bool = True,
        use_safety_model: bool = True,
        store_transcripts: bool = False,
        transcript_log_path=None,
    ):
        self.capture = AudioCaptureService(device=device)
        self.yamnet = YamNetAudioDetector(yamnet_model_path, device=device)
        self.loudness = LoudnessAnalyzer()
        self.transcriber = FasterWhisperTranscriber(
            whisper_model,
            device=transcription_device,
            compute_type=transcription_compute_type,
            language=transcription_language,
        )
        self.safety = BilingualSafetyClassifier(use_model=use_safety_model)
        self.display_transcript = display_transcript
        self.mask_unsafe_transcript = mask_unsafe_transcript
        self.store_transcripts = store_transcripts
        self._transcript_log_path = transcript_log_path or "outputs/transcript_log.csv"
        self._transcript_store_lock = RLock()
        self._transcript_store = TranscriptStore(self._transcript_log_path) if store_transcripts else None
        self._gate = EnergyUtteranceGate()
        self._gate_lock = RLock()
        # Keep several short, in-memory utterances so a second sentence is not discarded while
        # Whisper is still finishing the first one. Nothing in this queue is written as audio.
        self._transcription_queue: Queue[tuple[np.ndarray, int]] = Queue(maxsize=8)
        self._stop_event = Event()
        self._dispatch_worker: Thread | None = None
        self._transcription_worker: Thread | None = None
        self._state_lock = RLock()
        self._calibration_lock = RLock()
        self._latest_state = replace(
            SpeechAnalysisState.unavailable("Audio disabled"),
            transcript_display_enabled=self.display_transcript,
            transcript_recording_enabled=self.store_transcripts,
        )
        self._calibration_remaining_samples = 0
        self._calibration_parts: list[np.ndarray] = []
        self._transcription_queued_count = 0
        self._transcription_completed_count = 0
        self._transcription_drop_count = 0
        self._transcription_max_queue_depth = 0
        self._transcription_total_ms = 0.0
        self._transcription_max_ms = 0.0

    def start(self) -> bool:
        if self._dispatch_worker is not None:
            return True
        if not self.yamnet.start(capture_microphone=False):
            self._set_state(SpeechAnalysisState.unavailable(self.yamnet.get_latest_state().error))
            return False
        if not self.capture.start():
            self.yamnet.stop()
            self._set_state(SpeechAnalysisState.unavailable(self.capture.error))
            return False

        self._stop_event.clear()
        self._dispatch_worker = Thread(target=self._run_dispatch, name="audio-dispatch", daemon=True)
        self._transcription_worker = Thread(
            target=self._run_transcription, name="speech-transcription", daemon=True
        )
        self._dispatch_worker.start()
        self._transcription_worker.start()
        self._set_state(
            SpeechAnalysisState(
                audio_available=True,
                audio_top_label="Listening",
                transcript_display_enabled=self.display_transcript,
                transcript_recording_enabled=self.store_transcripts,
                transcription_status=f"Listening ({self.transcriber.runtime_summary})",
                error="",
            )
        )
        return True

    def stop(self) -> None:
        self.capture.stop()
        self.finish_current_utterance()
        self._stop_event.set()
        if self._dispatch_worker is not None:
            self._dispatch_worker.join(timeout=2.0)
        # Preserve an already queued consented transcript before its CSV is closed.
        if self._transcription_worker is not None:
            self._transcription_worker.join(timeout=30.0)
        self._dispatch_worker = None
        self._transcription_worker = None
        self.yamnet.stop()
        self._clear_transcription_queue()
        with self._transcript_store_lock:
            if self._transcript_store is not None:
                self._transcript_store.close()
            self._transcript_store = None
        with self._calibration_lock:
            self._calibration_remaining_samples = 0
            self._calibration_parts = []

    def calibrate_normal_voice(self, samples: np.ndarray) -> float:
        """Calibrate voice level from an explicit normal-speaking recording."""
        return self.loudness.calibrate(samples)

    def request_voice_calibration(self, duration_seconds: int = 5) -> bool:
        """Capture a short in-memory normal-speaking sample, then immediately discard it."""
        if self._dispatch_worker is None:
            return False
        with self._calibration_lock:
            self._calibration_remaining_samples = max(1, duration_seconds) * self.capture.sample_rate
            self._calibration_parts = []
        return True

    def set_transcript_display(self, enabled: bool) -> bool:
        """Explicitly control whether the short in-memory transcript is shown."""
        self.display_transcript = bool(enabled)
        with self._state_lock:
            state = self._latest_state
            self._latest_state = replace(
                state,
                transcript_display_enabled=self.display_transcript,
                transcript_visible=self.display_transcript and bool(state.transient_transcript),
            )
        return self.display_transcript

    def set_transcript_recording(self, enabled: bool) -> bool:
        """Enable or disable consented transcript CSV recording; raw audio is never recorded."""
        enabled = bool(enabled)
        with self._transcript_store_lock:
            if enabled and self._transcript_store is None:
                self._transcript_store = TranscriptStore(self._transcript_log_path)
            elif not enabled and self._transcript_store is not None:
                self._transcript_store.close()
                self._transcript_store = None
            self.store_transcripts = enabled
        with self._state_lock:
            self._latest_state = replace(
                self._latest_state,
                transcript_recording_enabled=enabled,
            )
        return enabled

    def export_transcripts(self, destination_path) -> tuple[bool, str]:
        """Explicitly export saved transcripts as readable text; never exports raw audio."""
        try:
            with self._transcript_store_lock:
                path = export_readable_transcript(self._transcript_log_path, destination_path)
            return True, str(path)
        except (OSError, ValueError) as exc:
            return False, str(exc)

    def finish_current_utterance(self) -> bool:
        """Queue the current in-memory speech segment without requiring trailing silence."""
        with self._gate_lock:
            utterance = self._gate.flush()
        if utterance is None or len(utterance) < 8_000:
            return False
        self._queue_transcription(utterance, int(time.monotonic() * 1000))
        return True

    def wait_for_pending_transcription(self, timeout_seconds: float = 30.0) -> bool:
        """Wait briefly for a queued transcription, such as before a smoke test exits."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            state = self.get_latest_state()
            if not state.transcription_pending and self._transcription_queue.empty():
                return True
            time.sleep(0.1)
        return False

    def get_latest_state(self) -> SpeechAnalysisState:
        with self._state_lock:
            state = self._latest_state
        if not state.transient_transcript or state.transcript_timestamp_ms <= 0:
            return state
        if int(time.monotonic() * 1000) - state.transcript_timestamp_ms > TRANSCRIPT_TTL_MS:
            return replace(state, transient_transcript="", transcript_visible=False)
        return state

    def runtime_statistics(self) -> dict[str, int | float | dict[str, int | float]]:
        with self._state_lock:
            transcription_count = self._transcription_completed_count
            transcription = {
                "queued_count": self._transcription_queued_count,
                "completed_count": transcription_count,
                "queue_drops": self._transcription_drop_count,
                "queue_depth": self._transcription_queue.qsize(),
                "max_queue_depth": self._transcription_max_queue_depth,
                "inference_mean_ms": round(self._transcription_total_ms / transcription_count, 4)
                if transcription_count
                else 0.0,
                "inference_max_ms": round(self._transcription_max_ms, 4),
            }
        return {
            "capture": self.capture.runtime_statistics(),
            "yamnet": self.yamnet.runtime_statistics(),
            "transcription": transcription,
        }

    def _run_dispatch(self) -> None:
        while not self._stop_event.is_set():
            chunk = self.capture.get()
            if chunk is None:
                continue
            self.yamnet.feed_audio(chunk.samples)
            self._collect_calibration_samples(chunk.samples)
            _, chunk_dbfs = LoudnessAnalyzer.rms_dbfs(chunk.samples)
            with self._gate_lock:
                _segment_active, utterance = self._gate.update(chunk.samples, chunk_dbfs)
                speech_active = self._gate.last_chunk_speech
            loudness_state = self.loudness.update(chunk.samples, speech_active=speech_active)
            audio_state = self.yamnet.get_latest_state()
            self._publish_audio_state(audio_state, loudness_state, speech_active)
            if utterance is not None and len(utterance) >= 8_000:
                self._queue_transcription(utterance, chunk.timestamp_ms)

    def _run_transcription(self) -> None:
        while True:
            try:
                samples, timestamp_ms = self._transcription_queue.get(timeout=0.2)
            except Empty:
                if self._stop_event.is_set():
                    return
                continue
            self._set_transcription_status("Transcribing", pending=True)
            started = time.perf_counter()
            try:
                transcript = self.transcriber.transcribe(samples)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with self._state_lock:
                    self._transcription_completed_count += 1
                    self._transcription_total_ms += elapsed_ms
                    self._transcription_max_ms = max(self._transcription_max_ms, elapsed_ms)
            if transcript.error or not transcript.text:
                if transcript.error:
                    self._update_analysis_error(transcript.error)
                    self._set_transcription_status("Transcription failed", pending=False)
                else:
                    self._set_transcription_status("No speech recognized", pending=False)
                continue
            safety = self.safety.analyze(transcript.text, language=transcript.language)
            transcript_saved = False
            save_error = ""
            with self._transcript_store_lock:
                if self._transcript_store is not None:
                    try:
                        self._transcript_store.append(
                            timestamp_ms=timestamp_ms,
                            language=transcript.language,
                            transcript=transcript.text,
                            safety_categories=safety.categories,
                            safety_confidence=max(safety.toxicity_confidence, safety.rule_confidence),
                        )
                        transcript_saved = True
                    except OSError as exc:
                        save_error = (
                            "Transcript was recognized but could not be saved. Close "
                            f"outputs/transcript_log.csv if it is open in Excel, then try again: {exc}"
                        )
            display_text = transcript.text
            if safety.flagged and self.mask_unsafe_transcript:
                display_text = "[unsafe cabin speech masked]"
            with self._state_lock:
                previous = self._latest_state
                self._latest_state = replace(
                    previous,
                    language=transcript.language,
                    language_probability=round(transcript.language_probability, 4),
                    transient_transcript=display_text,
                    transcript_visible=self.display_transcript and bool(display_text),
                    transcript_display_enabled=self.display_transcript,
                    transcript_recording_enabled=self.store_transcripts,
                    recorded_transcript_count=(
                        previous.recorded_transcript_count + int(transcript_saved)
                    ),
                    last_transcript_saved_ms=(timestamp_ms if transcript_saved else previous.last_transcript_saved_ms),
                    transcription_pending=False,
                    transcription_status="Transcript saved" if transcript_saved else "Transcript ready",
                    safety_flagged=safety.flagged,
                    safety_categories=safety.categories,
                    safety_confidence=max(safety.toxicity_confidence, safety.rule_confidence),
                    safety_model_available=safety.model_available,
                    timestamp_ms=timestamp_ms,
                    transcript_timestamp_ms=int(time.monotonic() * 1000),
                    analysis_error=save_error or safety.error or self.transcriber.runtime_warning,
                )

    def _queue_transcription(self, samples: np.ndarray, timestamp_ms: int) -> None:
        try:
            self._transcription_queue.put_nowait((samples, timestamp_ms))
            with self._state_lock:
                self._transcription_queued_count += 1
                self._transcription_max_queue_depth = max(
                    self._transcription_max_queue_depth,
                    self._transcription_queue.qsize(),
                )
            self._set_transcription_status("Queued for transcription", pending=True)
        except Full:
            with self._state_lock:
                self._transcription_drop_count += 1
            try:
                self._transcription_queue.get_nowait()
                self._transcription_queue.put_nowait((samples, timestamp_ms))
                with self._state_lock:
                    self._transcription_max_queue_depth = max(
                        self._transcription_max_queue_depth,
                        self._transcription_queue.qsize(),
                    )
                self._report_dropped_utterance()
            except (Empty, Full):
                pass

    def _collect_calibration_samples(self, samples: np.ndarray) -> None:
        with self._calibration_lock:
            if self._calibration_remaining_samples <= 0:
                return
            take = min(self._calibration_remaining_samples, len(samples))
            self._calibration_parts.append(np.asarray(samples[:take], dtype=np.float32).copy())
            self._calibration_remaining_samples -= take
            if self._calibration_remaining_samples > 0:
                return
            calibration_samples = np.concatenate(self._calibration_parts)
            self._calibration_parts = []
        self.loudness.calibrate(calibration_samples)

    def _publish_audio_state(
        self,
        audio_state: AudioState,
        loudness_state: LoudnessState,
        speech_active: bool,
    ) -> None:
        with self._state_lock:
            previous = self._latest_state
            self._latest_state = replace(
                previous,
                audio_available=audio_state.classifier_available,
                audio_top_label=audio_state.top_label,
                audio_top_confidence=audio_state.top_confidence,
                raised_voice_active=audio_state.raised_voice_active,
                horn_active=audio_state.horn_active,
                siren_active=audio_state.siren_active,
                music_active=audio_state.music_active,
                quiet_active=audio_state.quiet_active,
                speech_active=speech_active,
                voice_level=loudness_state.voice_level,
                loudness_dbfs=loudness_state.dbfs,
                loud_voice_active=loudness_state.loud_voice_active,
                timestamp_ms=loudness_state.timestamp_ms,
                error=audio_state.error,
            )

    def _update_analysis_error(self, error: str) -> None:
        with self._state_lock:
            self._latest_state = replace(self._latest_state, analysis_error=error)

    def _set_transcription_status(self, status: str, *, pending: bool) -> None:
        with self._state_lock:
            self._latest_state = replace(
                self._latest_state,
                transcription_status=status,
                transcription_pending=pending,
            )

    def _report_dropped_utterance(self) -> None:
        with self._state_lock:
            self._latest_state = replace(
                self._latest_state,
                dropped_utterance_count=self._latest_state.dropped_utterance_count + 1,
                transcription_status="Queue full: an older utterance was dropped",
                transcription_pending=True,
                analysis_error="Transcription could not keep up with incoming speech",
            )

    def _set_state(self, state: SpeechAnalysisState) -> None:
        with self._state_lock:
            self._latest_state = state

    def _clear_transcription_queue(self) -> None:
        while True:
            try:
                self._transcription_queue.get_nowait()
            except Empty:
                return
