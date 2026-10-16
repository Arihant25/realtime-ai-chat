"""Microphone capture and a gapless, interruptible speaker queue."""

from __future__ import annotations

import collections
import queue
import threading
import time

import numpy as np
import sounddevice as sd


class Microphone:
    """16 kHz mono float32 frames of a fixed size, delivered through a queue."""

    def __init__(self, sample_rate: int, frame: int, device=None):
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32", blocksize=frame,
            device=device, callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        self.frames.put(indata[:, 0].copy())

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()


FADE = 120  # 5 ms at 24 kHz
_RAMP = np.linspace(0.0, 1.0, FADE, dtype=np.float32)


class Player:
    """Plays queued float32 chunks back to back.

    If generation can't keep up with playback, it pauses and waits until
    `prebuffer_s` of audio is queued again (or the stream is marked finished)
    instead of crackling through underruns.
    """

    def __init__(self, sample_rate: int, prebuffer_s: float = 0.3, device=None,
                 rebuffer_s: float = 0.4):
        self.prebuffer = int(prebuffer_s * sample_rate)
        self.rebuffer = int(max(rebuffer_s, prebuffer_s) * sample_rate)
        self._stalled = False
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._offset = 0
        self._queued = 0
        self._holding = True
        self._finished = False
        self._gated = False
        self._lock = threading.Lock()
        self.last_sound = 0.0
        # Monotonic sample counters: chunk k plays once played_total passes its fed offset.
        # Used to sync on-screen words with what is actually coming out of the speakers.
        self.fed_total = 0
        self.played_total = 0
        self.level = 0.0  # RMS of the last output block
        self.underflows = 0  # driver ran dry: Python callback missed its deadline (clicks)
        self.stalls = 0  # our queue ran dry mid-reply: generation fell behind (gaps)
        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32", device=device,
            callback=self._callback, blocksize=1024,
        )
        self.stream.start()

    def feed(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(chunk.astype(np.float32, copy=False))
            self._queued += len(chunk)
            self.fed_total += len(chunk)
            self._finished = False

    def finish(self) -> None:
        """No more audio for this reply: play out whatever is queued."""
        with self._lock:
            self._finished = True

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._offset = 0
            self._queued = 0
            self.fed_total = self.played_total  # dropped audio never plays
            self._holding = True
            self._finished = False
            self._stalled = False

    def set_gate(self, gated: bool) -> None:
        """While gated, audio keeps queueing but nothing plays (speculative replies)."""
        with self._lock:
            self._gated = gated

    def busy(self) -> bool:
        with self._lock:
            return self._queued > 0 and not self._gated

    def _callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        out.fill(0)
        self.level = 0.0
        if status.output_underflow:
            self.underflows += 1
        with self._lock:
            if self._gated:
                return
            resumed = False
            if self._holding:
                # after a stall, wait for a bigger cushion so it doesn't stutter right away
                need = self.rebuffer if self._stalled else self.prebuffer
                if self._queued >= need or (self._finished and self._queued):
                    self._holding = False
                    resumed = self._stalled
                else:
                    return
            written = 0
            while written < frames and self._chunks:
                chunk = self._chunks[0]
                n = min(frames - written, len(chunk) - self._offset)
                out[written : written + n] = chunk[self._offset : self._offset + n]
                written += n
                self._offset += n
                self._queued -= n
                if self._offset >= len(chunk):
                    self._chunks.popleft()
                    self._offset = 0
            if written and resumed:  # fade back in after a stall instead of a hard restart
                n = min(FADE, written)
                out[:n] *= _RAMP[:n]
                self._stalled = False
            if not self._chunks:
                self._holding = True  # underrun or done: rebuffer before resuming
                if written and not self._finished:
                    # generation fell behind mid-reply: fade out rather than cut (a cut clicks)
                    self.stalls += 1
                    self._stalled = True
                    n = min(FADE, written)
                    out[written - n : written] *= _RAMP[:n][::-1]
            if written:
                self.last_sound = time.monotonic()
                self.played_total += written
                self.level = float(np.sqrt(np.mean(out[:written] ** 2)))

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()
