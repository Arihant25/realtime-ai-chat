"""Run the real turn-taking loop on WAV files instead of a mic, with a virtual speaker.

Measures voice-to-voice latency (your last word -> first audible sample) and
playback stalls, and checks that a mid-sentence pause is retracted, not answered.

    uv run python bench/simulate.py --speech-dir <dir with .wav/.txt> --llm-model Qwen3_5-4B-IQ4_XS
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from bench.bench import load_wav
from realtime_chat.kokoro_tts import KokoroTTS
from realtime_chat.llm import LLMClient, system_prompt
from realtime_chat.main import MODELS_DIR, Conversation, Listener
from realtime_chat.stt import Parakeet
from realtime_chat.tts import SAMPLE_RATE as TTS_RATE
from realtime_chat.tts import OrpheusTTS
from realtime_chat.vad import FRAME, SAMPLE_RATE, SileroVAD, UtteranceSegmenter, ensure_silero


class VirtualPlayer:
    """Same interface as audio.Player; consumes audio at real-time rate and logs timing."""

    def __init__(self, prebuffer_s: float):
        self.prebuffer = int(prebuffer_s * TTS_RATE)
        self.lock = threading.Lock()
        self.queued = 0
        self.gated = False
        self.holding = True
        self.finished = False
        self.last_sound = 0.0
        self.first_sound: float | None = None
        self.stalls = 0
        self.played = 0
        self.fed_total = self.played_total = 0
        self.level = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def feed(self, chunk):
        with self.lock:
            self.queued += len(chunk)
            self.fed_total += len(chunk)
            self.finished = False

    def finish(self):
        with self.lock:
            self.finished = True

    def clear(self):
        with self.lock:
            self.queued, self.holding, self.finished = 0, True, False
            self.fed_total = self.played_total

    def set_gate(self, gated):
        with self.lock:
            self.gated = gated

    def busy(self):
        with self.lock:
            return self.queued > 0 and not self.gated

    def _run(self):
        block = 1024
        while True:
            time.sleep(block / TTS_RATE)
            with self.lock:
                if self.gated:
                    continue
                if self.holding:
                    if self.queued >= self.prebuffer or (self.finished and self.queued):
                        self.holding = False
                    else:
                        continue
                n = min(block, self.queued)
                self.level = 0.05 if n else 0.0
                if n:
                    self.queued -= n
                    self.played += n
                    self.played_total += n
                    self.last_sound = time.monotonic()
                    self.first_sound = self.first_sound or self.last_sound
                if self.queued == 0:
                    self.holding = True
                    if not self.finished:
                        self.stalls += 1


def run_turn(listener, player, convo, audio, gap_s=2.5, extra=None):
    """Feed audio at real-time pace; return (latency_s, stalls)."""
    player.first_sound, player.stalls = None, 0
    parts = [audio] if extra is None else [audio, np.zeros(int(0.4 * SAMPLE_RATE), np.float32), extra]
    stream = np.concatenate(parts + [np.zeros(int(gap_s * SAMPLE_RATE), np.float32)])
    last = parts[-1]
    loud = np.flatnonzero(np.abs(last) > 0.01)
    speech_end_idx = len(stream) - int(gap_s * SAMPLE_RATE) - len(last) + (loud[-1] if len(loud) else 0)
    t0 = time.monotonic()
    speech_end_t = None
    for i in range(0, len(stream) - FRAME + 1, FRAME):
        target = t0 + (i + FRAME) / SAMPLE_RATE
        time.sleep(max(0, target - time.monotonic()))
        if speech_end_t is None and i + FRAME >= speech_end_idx:
            speech_end_t = time.monotonic()
        listener.handle(stream[i : i + FRAME])
    while convo.active():  # let the reply finish
        time.sleep(0.05)
    time.sleep(0.6)  # past the half-duplex echo tail, so the next turn isn't clipped
    latency = player.first_sound - speech_end_t if player.first_sound else float("nan")
    return latency, player.stalls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speech-dir", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:1337/v1")
    p.add_argument("--llm-model", default="Qwen3_5-4B-IQ4_XS")
    p.add_argument("--tts", default="kokoro", choices=["kokoro", "kokoro-gpu", "orpheus"])
    p.add_argument("--tts-model", default="Orpheus-3b-FT-Q4_K_M")
    p.add_argument("--early-ms", type=int, default=250)
    p.add_argument("--silence-ms", type=int, default=600)
    p.add_argument("--prebuffer", type=float, default=0.3)
    p.add_argument("--tts-parallel", type=int, default=2)
    args = p.parse_args()

    wavs = sorted(args.speech_dir.glob("*.wav"))
    clips = [load_wav(w) for w in wavs]
    stt = Parakeet()
    segmenter = UtteranceSegmenter(SileroVAD(ensure_silero(MODELS_DIR)),
                                   min_silence_ms=args.silence_ms, early_silence_ms=args.early_ms)
    player = VirtualPlayer(args.prebuffer)
    llm = LLMClient(args.base_url, args.llm_model)
    if args.tts.startswith("kokoro"):
        tts = KokoroTTS(MODELS_DIR, gpu=args.tts == "kokoro-gpu")
    else:
        tts = OrpheusTTS(args.base_url, args.tts_model)
    convo = Conversation(llm, tts, player, system_prompt(sound_tags=args.tts == "orpheus"),
                         tts_parallel=args.tts_parallel)
    listener = Listener(segmenter, stt, convo, player)

    lat, stalls = [], []
    for w, clip in zip(wavs, clips):
        convo.history = convo.history[:1]
        l, s = run_turn(listener, player, convo, clip)
        lat.append(l)
        stalls.append(s)
        print(f"  {w.stem}: voice-to-voice {l*1000:.0f} ms, {s} stall(s); "
              f"AI said: {convo.history[-1]['content'][:80]!r}")

    # A 400 ms mid-sentence pause (between --early-ms and --silence-ms) must not be answered.
    convo.history = convo.history[:1]
    run_turn(listener, player, convo, clips[0][: len(clips[0]) // 2], extra=clips[0][len(clips[0]) // 2 :])
    users = [m["content"] for m in convo.history if m["role"] == "user"]
    print(f"  pause test: {len(users)} user turn(s) recorded: {users}")

    print(f"\ntts={args.tts} llm={args.llm_model} early={args.early_ms} silence={args.silence_ms} prebuffer={args.prebuffer} "
          f"parallel={args.tts_parallel}: median voice-to-voice "
          f"{statistics.median(lat)*1000:.0f} ms, stalls/turn {statistics.mean(stalls):.1f}, "
          f"pause test {'PASS' if len(users) == 1 else 'FAIL'}")


if __name__ == "__main__":
    main()
