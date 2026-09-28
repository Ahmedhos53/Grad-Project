"""Local Arabic/English transcription with Faster-Whisper."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import time

import numpy as np

from src.audio.cuda_runtime import configure_project_cuda_runtime


SUPPORTED_BILINGUAL_LANGUAGES = {"ar", "en"}


@dataclass(frozen=True)
class TranscriptionRuntime:
    """Concrete Faster-Whisper runtime selected for the current machine."""

    device: str
    compute_type: str


def select_transcription_runtime(
    device: str = "auto",
    compute_type: str = "auto",
    *,
    cuda_available: bool,
) -> TranscriptionRuntime:
    """Resolve user preferences without loading a Whisper model.

    CUDA uses FP16 by default for speed and accuracy. CPU uses INT8 to keep the
    local fallback practical on a laptop. Explicit compute-type choices are
    preserved so advanced users can tune their own environment.
    """

    requested_device = str(device or "auto").strip().lower()
    requested_compute_type = str(compute_type or "auto").strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Transcription device must be one of: auto, cpu, cuda")

    selected_device = "cuda" if requested_device == "auto" and cuda_available else requested_device
    if selected_device == "auto":
        selected_device = "cpu"
    selected_compute_type = requested_compute_type
    if selected_compute_type == "auto":
        selected_compute_type = "float16" if selected_device == "cuda" else "int8"
    return TranscriptionRuntime(selected_device, selected_compute_type)


@dataclass(frozen=True)
class TranscriptResult:
    text: str = ""
    language: str = "unknown"
    language_probability: float = 0.0
    average_log_probability: float = 0.0
    processing_latency_ms: int = 0
    error: str = ""


class FasterWhisperTranscriber:
    """Lazy local multilingual Whisper model; no audio is written to disk."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int | None = None,
        language: str | None = None,
        initial_prompt: str | None = None,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.cpu_threads = cpu_threads or max(2, min(4, os.cpu_count() or 2))
        self.language = language
        self.initial_prompt = initial_prompt
        self._model = None
        self._error = ""
        self._runtime_warning = ""
        self.cuda_runtime_dir = configure_project_cuda_runtime()
        self._runtime = select_transcription_runtime(
            device,
            compute_type,
            cuda_available=self._cuda_available(),
        )

    @staticmethod
    def _cuda_available() -> bool:
        """Return whether CTranslate2 can see an NVIDIA CUDA device."""

        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    @property
    def device(self) -> str:
        """The active or planned device, retained for concise status reporting."""

        return self._runtime.device

    @property
    def compute_type(self) -> str:
        return self._runtime.compute_type

    @property
    def runtime_summary(self) -> str:
        summary = f"{self.model_name} on {self.device} ({self.compute_type})"
        return f"{summary}; {self._runtime_warning}" if self._runtime_warning else summary

    @property
    def runtime_warning(self) -> str:
        return self._runtime_warning

    @property
    def error(self) -> str:
        return self._error

    def preload(self) -> bool:
        """Download/load the selected model before live microphone capture begins."""

        return self._ensure_model()

    def transcribe(self, samples: np.ndarray, *, sample_rate: int = 16_000) -> TranscriptResult:
        if sample_rate != 16_000:
            return TranscriptResult(error="Transcription expects 16 kHz mono audio")
        if not self._ensure_model():
            return TranscriptResult(error=self._error)

        start = time.monotonic()
        try:
            return self._transcribe_loaded_model(samples, start)
        except Exception as exc:
            if self.device != "cuda":
                return TranscriptResult(error=f"Transcription failed: {exc}")

            gpu_error = str(exc)
            self._model = None
            self._runtime = TranscriptionRuntime("cpu", "int8")
            self._runtime_warning = (
                "GPU transcription failed during decoding; using CPU INT8 instead "
                f"({gpu_error})"
            )
            if not self._ensure_model():
                return TranscriptResult(error=self._error)
            try:
                return self._transcribe_loaded_model(samples, start)
            except Exception as fallback_exc:
                return TranscriptResult(
                    error=(
                        "Transcription failed on CUDA and CPU fallback: "
                        f"CUDA error: {gpu_error}; CPU error: {fallback_exc}"
                    )
                )

    def _transcribe_loaded_model(self, samples: np.ndarray, start: float) -> TranscriptResult:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        result = self._decode_once(
            values,
            language=self.language,
            initial_prompt=self._effective_initial_prompt(),
        )

        # Automatic language detection can confuse short Arabic speech with
        # another Arabic-script language (especially Urdu) or occasionally
        # French. CabInspector supports Arabic and English only, so retry an
        # unsupported result in both supported modes and select the stronger
        # transcript. The audio remains in memory and is not persisted.
        if self.language is None and result.language not in SUPPORTED_BILINGUAL_LANGUAGES:
            arabic = self._decode_once(
                values,
                language="ar",
                initial_prompt=None,
            )
            english = self._decode_once(values, language="en", initial_prompt=None)
            preferred_script = self._preferred_script(result.text)
            result = max(
                (arabic, english),
                key=lambda candidate: self._candidate_score(candidate, preferred_script),
            )

        return replace(
            result,
            processing_latency_ms=int((time.monotonic() - start) * 1000),
        )

    def _decode_once(
        self,
        samples: np.ndarray,
        *,
        language: str | None,
        initial_prompt: str | None,
    ) -> TranscriptResult:
        segments, info = self._model.transcribe(
            samples,
            beam_size=5,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        average_log_probability = (
            sum(segment.avg_logprob for segment in segments) / len(segments)
            if segments
            else 0.0
        )
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", "unknown"),
            language_probability=float(getattr(info, "language_probability", 0.0)),
            average_log_probability=float(average_log_probability),
        )

    @staticmethod
    def _preferred_script(text: str) -> str | None:
        if any("\u0600" <= character <= "\u06ff" for character in text):
            return "ar"
        if any(("a" <= character.lower() <= "z") for character in text):
            return "en"
        return None

    @staticmethod
    def _candidate_score(result: TranscriptResult, preferred_script: str | None) -> float:
        if not result.text:
            return float("-inf")
        script_bonus = 0.15 if result.language == preferred_script else 0.0
        return result.average_log_probability + script_bonus

    def _effective_initial_prompt(self) -> str | None:
        # Automatic prompts can be copied into short utterances, producing words
        # that were never spoken. Keep the default decoder prompt-free. A caller
        # may still explicitly supply a domain prompt after evaluating it.
        return self.initial_prompt

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._error:
            return False
        try:
            self._model = self._create_model(self._runtime)
            return True
        except Exception as exc:
            if self.device != "cuda":
                self._error = f"Whisper model unavailable: {exc}"
                return False

            gpu_error = str(exc)
            self._runtime = TranscriptionRuntime("cpu", "int8")
            self._runtime_warning = (
                "GPU transcription could not start; using CPU INT8 instead "
                f"({gpu_error})"
            )
            try:
                self._model = self._create_model(self._runtime)
                return True
            except Exception as fallback_exc:
                self._error = (
                    "Whisper model unavailable on CUDA and CPU fallback failed: "
                    f"CUDA error: {gpu_error}; CPU error: {fallback_exc}"
                )
                return False

    def _create_model(self, runtime: TranscriptionRuntime):
        # The standard CDN path has proved more reliable than Hugging Face Xet on
        # this Windows setup. Advanced users can opt back into Xet explicitly.
        if os.environ.get("CABINSPECTOR_HUGGINGFACE_USE_XET", "").strip().lower() not in {
            "1",
            "true",
            "yes",
        }:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.model_name,
            device=runtime.device,
            compute_type=runtime.compute_type,
            cpu_threads=self.cpu_threads,
        )
