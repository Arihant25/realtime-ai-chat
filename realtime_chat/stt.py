"""Speech-to-text with NVIDIA Parakeet TDT 0.6B v2 via its ONNX port (no NeMo)."""

from __future__ import annotations

import numpy as np
import onnx_asr
import onnxruntime as ort

PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"  # istupakov/parakeet-tdt-0.6b-v2-onnx on HF

# onnx-asr model names offered in the web UI (weights download on first use)
MODELS = {
    "nemo-parakeet-tdt-0.6b-v2": "Parakeet TDT 0.6B v2 (English, best)",
    "nemo-parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3 (25 languages)",
    "nemo-parakeet-ctc-0.6b": "Parakeet CTC 0.6B (English, faster decode)",
    "nemo-canary-1b-v2": "Canary 1B v2 (multilingual, slower)",
    "whisper-base": "Whisper base (small, multilingual)",
}


class Parakeet:
    """Any onnx-asr model; named for the default. CPU uses int8 weights (~650 MB RAM for
    Parakeet). GPU uses fp32 (~2.5 GB VRAM) because int8 ONNX models mostly fall back
    to CPU kernels on GPU providers."""

    def __init__(self, gpu: bool = False, model: str = PARAKEET_MODEL):
        self.name = model
        if gpu:
            available = ort.get_available_providers()
            gpu_ep = next((p for p in ("CUDAExecutionProvider", "DmlExecutionProvider")
                           if p in available), None)
            if gpu_ep is None:
                raise SystemExit(f"--stt-gpu: no GPU execution provider in onnxruntime ({available})")
            providers, quantization = [gpu_ep, "CPUExecutionProvider"], None
        else:
            providers, quantization = ["CPUExecutionProvider"], "int8"
        # Weights download from Hugging Face on first use and are cached afterwards.
        try:
            self.model = onnx_asr.load_model(model, quantization=quantization, providers=providers)
        except Exception:
            if quantization is None:
                raise
            # not every model ships int8 weights; fall back to full precision
            self.model = onnx_asr.load_model(model, providers=providers)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return self.model.recognize(audio.astype(np.float32), sample_rate=sample_rate).strip()
