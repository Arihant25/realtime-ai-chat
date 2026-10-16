"""Does the server overlap concurrent Orpheus requests? Compares 1 vs N parallel streams."""

import argparse
import threading
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://127.0.0.1:1337/v1")
p.add_argument("--model", default="Orpheus-3b-FT-Q4_K_M")
p.add_argument("-n", type=int, default=2)
args = p.parse_args()


def run(out, i):
    r = requests.post(f"{args.base_url}/completions", json={
        "model": args.model, "prompt": "<|audio|>tara: The quick brown fox jumps over the lazy dog "
        f"and then takes a long nap in the warm afternoon sun, number {i}.<|eot_id|>",
        "max_tokens": 300, "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1,
        "ignore_eos": True}, timeout=300).json()
    out[i] = r["timings"]["predicted_n"]


for n in (1, args.n):
    out = {}
    t = time.perf_counter()
    ths = [threading.Thread(target=run, args=(out, i)) for i in range(n)]
    [th.start() for th in ths]
    [th.join() for th in ths]
    wall = time.perf_counter() - t
    tokens = sum(out.values())
    print(f"{n} stream(s): {tokens} tokens in {wall:.2f}s = {tokens / wall:.0f} tok/s total "
          f"({tokens / wall / 82:.2f}x real-time audio)")
