"""Streaming Silero VAD (ONNX) and an utterance segmenter built on it."""

from __future__ import annotations

import collections
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
FRAME = 512  # samples per VAD step at 16 kHz (32 ms), fixed by Silero v5
CONTEXT = 64  # Silero v5 prepends the tail of the previous frame
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def ensure_silero(models_dir: Path) -> Path:
    path = models_dir / "silero_vad.onnx"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Silero VAD -> {path}")
        urllib.request.urlretrieve(SILERO_URL, path)
    return path


class SileroVAD:
    """Frame-by-frame speech probability, keeping the model's recurrent state."""

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame.reshape(1, -1)], axis=1)
        out, self._state = self.session.run(
            None,
            {"input": x, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)},
        )
        self._context = x[:, -CONTEXT:]
        return float(out[0][0])


class UtteranceSegmenter:
    """Turns a stream of 512-sample frames into complete utterances.

    Speech starts when probability crosses `threshold`; it ends after
    `min_silence_ms` below `threshold - 0.15`. A short pre-roll keeps the
    first syllable that arrived before the VAD fired.
    """

    def __init__(
        self,
        vad: SileroVAD,
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        early_silence_ms: int = 250,
        min_speech_ms: int = 250,
        pre_roll_ms: int = 300,
        max_utterance_s: float = 30.0,
    ):
        frame_ms = FRAME * 1000 / SAMPLE_RATE
        self.vad = vad
        self.on = threshold
        self.off = max(threshold - 0.15, 0.05)
        self.silence_frames = int(min_silence_ms / frame_ms)
        # early pause is unreachable (disabled) when it isn't shorter than the end-of-turn silence
        self.early_frames = (int(early_silence_ms / frame_ms) if early_silence_ms < min_silence_ms
                             else self.silence_frames + 1)
        self.min_speech_frames = int(min_speech_ms / frame_ms)
        self.max_frames = int(max_utterance_s * 1000 / frame_ms)
        self.pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=int(pre_roll_ms / frame_ms)
        )
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.pre_roll.clear()
        self.frames: list[np.ndarray] = []
        self.in_speech = False
        self.speech_frames = 0
        self.silent_run = 0

    def push(self, frame: np.ndarray) -> tuple[str | None, np.ndarray | None]:
        """Feed one frame. Returns (event, audio). Events:

        'start'  speech began
        'pause'  `early_silence_ms` of silence: audio so far, a likely (not certain) end
        'resume' speech came back after a 'pause'
        'end'    `min_silence_ms` of silence: the finished utterance (None if too short)
        """
        prob = self.vad(frame)
        if not self.in_speech:
            self.pre_roll.append(frame)
            if prob >= self.on:
                self.in_speech = True
                self.frames = list(self.pre_roll)
                self.pre_roll.clear()
                self.speech_frames = 1
                self.silent_run = 0
                return "start", None
            return None, None

        self.frames.append(frame)
        enough = self.speech_frames >= self.min_speech_frames
        if prob >= self.off:
            self.speech_frames += 1
            paused = self.silent_run >= self.early_frames
            self.silent_run = 0
            if paused and enough:
                return "resume", None
        else:
            self.silent_run += 1

        if self.silent_run >= self.silence_frames or len(self.frames) >= self.max_frames:
            audio = np.concatenate(self.frames)
            self.in_speech = False
            self.frames = []
            return "end", audio if enough else None
        if self.silent_run == self.early_frames and enough:
            return "pause", np.concatenate(self.frames)
        return None, None
