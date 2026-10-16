"""Kokoro-82M TTS via kokoro-onnx: a small non-autoregressive model, many times faster
than real time even on CPU, and it leaves the GPU to the LLM.

Same interface as OrpheusTTS: synthesize(text, cancel) yields float32 24 kHz chunks.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro

SAMPLE_RATE = 24000
# American (a) / British (b), female (f) / male (m). af_heart is the best-rated voice.
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_michael", "am_fenrir",
          "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable"]
_TAGS = re.compile(r"<\w+>")  # Orpheus-style sound tags; Kokoro would read them out
HF_REPO = "fastrtc/kokoro-onnx"  # mirrors the kokoro-onnx GitHub release files


def ensure_file(models_dir: Path, name: str) -> str:
    """Use models/<name> if present, else download it (cached by huggingface_hub)."""
    local = models_dir / name
    if local.exists():
        return str(local)
    return hf_hub_download(HF_REPO, name)


class KokoroTTS:
    def __init__(self, models_dir: Path, voice: str = "af_heart", speed: float = 1.0,
                 model_file: str = "kokoro-v1.0.onnx", gpu: bool = False, threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if gpu:
            # CUDA works; DirectML can't run this export (ConvTranspose in the F0 predictor fails)
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("Kokoro on GPU needs the CUDA build of onnxruntime")
            providers.insert(0, "CUDAExecutionProvider")
        session = ort.InferenceSession(ensure_file(models_dir, model_file), sess_options=opts,
                                       providers=providers)
        self.kokoro = Kokoro.from_session(session, ensure_file(models_dir, "voices-v1.0.bin"))
        available = self.kokoro.get_voices()
        if voice not in available:
            raise ValueError(f"Unknown Kokoro voice {voice!r}; choose from {', '.join(available)}")
        self.voice = voice
        self.speed = speed
        self.device = session.get_providers()[0]
        self._lock = None

    def synthesize(self, text: str, cancel: threading.Event) -> Iterator[np.ndarray]:
        text = re.sub(r"\s+", " ", _TAGS.sub("", text)).strip()
        if not text or cancel.is_set():
            return
        if self._lock:
            with self._lock:
                audio, _ = self.kokoro.create(text, voice=self.voice, speed=self.speed)
        else:
            audio, _ = self.kokoro.create(text, voice=self.voice, speed=self.speed)
        if cancel.is_set():
            return
        yield audio.astype(np.float32)
        # Kokoro trims silence, so add the natural gap the chunk boundary would have had
        gap = 0.08 if text.endswith((",", ";", ":", "—")) else 0.2
        yield np.zeros(int(gap * SAMPLE_RATE), np.float32)
