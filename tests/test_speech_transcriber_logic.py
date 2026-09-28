import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.audio.speech_transcriber import FasterWhisperTranscriber, select_transcription_runtime


class SpeechTranscriberLogicTests(unittest.TestCase):
    def test_auto_runtime_prefers_cuda_when_available(self):
        runtime = select_transcription_runtime("auto", "auto", cuda_available=True)

        self.assertEqual(runtime.device, "cuda")
        self.assertEqual(runtime.compute_type, "float16")

    def test_auto_runtime_uses_cpu_int8_when_cuda_is_unavailable(self):
        runtime = select_transcription_runtime("auto", "auto", cuda_available=False)

        self.assertEqual(runtime.device, "cpu")
        self.assertEqual(runtime.compute_type, "int8")

    def test_explicit_compute_type_is_preserved(self):
        runtime = select_transcription_runtime("cuda", "int8_float16", cuda_available=True)

        self.assertEqual(runtime.device, "cuda")
        self.assertEqual(runtime.compute_type, "int8_float16")

    def test_invalid_device_is_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            select_transcription_runtime("tpu", "auto", cuda_available=False)

    def test_cuda_load_failure_falls_back_to_cpu_int8(self):
        calls = []

        class FakeWhisperModel:
            def __init__(self, model_name, *, device, compute_type, cpu_threads):
                calls.append((model_name, device, compute_type, cpu_threads))
                if device == "cuda":
                    raise RuntimeError("CUDA libraries are unavailable")

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.object(FasterWhisperTranscriber, "_cuda_available", return_value=True):
            transcriber = FasterWhisperTranscriber("small", device="auto")
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            self.assertTrue(transcriber.preload())

        self.assertEqual(
            [(model, device, compute) for model, device, compute, _ in calls],
            [("small", "cuda", "float16"), ("small", "cpu", "int8")],
        )
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(transcriber.compute_type, "int8")
        self.assertIn("GPU transcription could not start", transcriber.runtime_warning)

    def test_cuda_decode_failure_retries_the_utterance_on_cpu(self):
        calls = []

        class FailingSegments:
            def __iter__(self):
                raise RuntimeError("CUDA decoder could not run")

        class FakeWhisperModel:
            def __init__(self, model_name, *, device, compute_type, cpu_threads):
                self.device = device
                calls.append((model_name, device, compute_type, cpu_threads))

            def transcribe(self, *_args, **_kwargs):
                info = SimpleNamespace(language="ar", language_probability=1.0)
                return (FailingSegments() if self.device == "cuda" else []), info

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.object(FasterWhisperTranscriber, "_cuda_available", return_value=True):
            transcriber = FasterWhisperTranscriber("small", device="auto")
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            result = transcriber.transcribe(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(result.error, "")
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(
            [(model, device, compute) for model, device, compute, _ in calls],
            [("small", "cuda", "float16"), ("small", "cpu", "int8")],
        )
        self.assertIn("failed during decoding", transcriber.runtime_warning)

    def test_bilingual_auto_mode_uses_no_automatic_prompt(self):
        received_prompts = []

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, *_args, **kwargs):
                received_prompts.append(kwargs["initial_prompt"])
                return [], SimpleNamespace(language="ar", language_probability=1.0)

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.object(FasterWhisperTranscriber, "_cuda_available", return_value=False):
            transcriber = FasterWhisperTranscriber("small", language=None)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            result = transcriber.transcribe(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(result.error, "")
        self.assertEqual(received_prompts, [None])

    def test_forced_arabic_mode_uses_no_automatic_prompt(self):
        received_prompts = []

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, *_args, **kwargs):
                received_prompts.append(kwargs["initial_prompt"])
                return [], SimpleNamespace(language="ar", language_probability=1.0)

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.object(FasterWhisperTranscriber, "_cuda_available", return_value=False):
            transcriber = FasterWhisperTranscriber("small", language="ar")
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            result = transcriber.transcribe(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(result.error, "")
        self.assertEqual(received_prompts, [None])

    def test_bilingual_mode_retries_unsupported_detected_language(self):
        received_languages = []

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, *_args, **kwargs):
                language = kwargs["language"]
                received_languages.append(language)
                if language is None:
                    segments = [SimpleNamespace(text="\u0643\u0644\u0627\u0645 \u062e\u0637\u0623", avg_logprob=-0.2)]
                    return segments, SimpleNamespace(language="ur", language_probability=0.8)
                if language == "ar":
                    segments = [SimpleNamespace(text="\u0643\u0644\u0628 \u062d\u0645\u0627\u0631 \u063a\u0628\u064a", avg_logprob=-0.5)]
                    return segments, SimpleNamespace(language="ar", language_probability=1.0)
                segments = [SimpleNamespace(text="incorrect English", avg_logprob=-0.4)]
                return segments, SimpleNamespace(language="en", language_probability=1.0)

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.object(FasterWhisperTranscriber, "_cuda_available", return_value=False):
            transcriber = FasterWhisperTranscriber("small", language=None)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            result = transcriber.transcribe(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(received_languages, [None, "ar", "en"])
        self.assertEqual(result.language, "ar")
        self.assertEqual(result.text, "\u0643\u0644\u0628 \u062d\u0645\u0627\u0631 \u063a\u0628\u064a")


if __name__ == "__main__":
    unittest.main()
