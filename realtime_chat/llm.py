"""Streaming chat with a local OpenAI-compatible server (Jan), split into speakable sentences."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator

import requests

def system_prompt(sound_tags: bool = True) -> str:
    """`sound_tags`: offer Orpheus's <laugh>-style tags (Kokoro can't voice them)."""
    return (
        "You are a friendly voice assistant in a real-time spoken conversation. "
        "Everything you write is read aloud by a text-to-speech engine, so reply in plain "
        "conversational sentences: no markdown, no lists, no code blocks, no emojis, and spell "
        "out symbols and abbreviations the way you would say them. Keep answers short, usually "
        "one to three sentences, unless the user asks for more. "
        + ("You may sparingly use these sound tags where they fit naturally: "
           "<laugh>, <chuckle>, <sigh>, <gasp>. " if sound_tags else "")
        # A short first clause lets speech start ~1-2 s sooner (it is synthesized on its own).
        # Several varied examples matter: with one example, small models repeat it verbatim.
        + "Open with one short reaction that fits what the user just said, ending in a comma, "
        "for example \"Ooh, fun one,\" or \"Oh no, that sounds rough,\" or \"Good question,\" or "
        "\"Sure,\". Vary it every time and never use more than one."
    )


SYSTEM_PROMPT = system_prompt()

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
# A sentence ends at . ! ? (optionally followed by quotes/brackets) and then whitespace.
_SENTENCE_END = re.compile(r"""(?<=[.!?…])["')\]]*\s+""")
_CLAUSE_END = re.compile(r"""(?<=[.!?…,;:—])["')\]]*\s+""")
_ONLY_TAGS = re.compile(r"(?:<\w+>\s*)*")


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.7,
                 max_tokens: int = 400):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        """Yield content deltas. Thinking is disabled and any <think> block is dropped."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with requests.post(self.url, headers=self.headers, json=payload, stream=True,
                           timeout=120) as r:
            r.raise_for_status()
            raw = ""
            emitted = 0
            for line in r.iter_lines(decode_unicode=True):
                if cancel.is_set():
                    return
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choice = json.loads(data)["choices"][0]
                # reasoning_content (if the server splits it out) is never spoken
                raw += choice.get("delta", {}).get("content") or ""
                visible = _THINK_RE.sub("", raw)
                if "<think>" in raw and "</think>" not in raw:
                    continue  # still inside a thinking block
                if len(visible) > emitted:
                    yield visible[emitted:]
                    emitted = len(visible)


def clean_for_speech(text: str) -> str:
    """Strip markdown and other things a TTS engine would read literally."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"^\s*>+\s?", "", text, flags=re.MULTILINE)  # blockquotes; keep <laugh> tags
    text = re.sub(r"[*_`#|~]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    # drop emojis / pictographs
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def sentences(deltas: Iterator[str], min_chars: int = 12, max_chars: int = 220,
              first_clause: bool = True) -> Iterator[str]:
    """Group streamed text into sentences, as early as possible, for TTS.

    With `first_clause`, the very first chunk may end at a comma/colon/dash, so
    speech can start after a short opening phrase instead of a whole sentence.
    """
    buf = ""
    carry = ""  # a chunk that was only sound tags, e.g. "<chuckle>", rides on the next one
    first = first_clause
    for delta in deltas:
        buf += delta
        while True:
            # paragraph breaks always end a chunk
            nl = buf.find("\n")
            m = (_CLAUSE_END.search(buf, 6) if first else _SENTENCE_END.search(buf, min_chars))
            cut = None
            if nl != -1 and (m is None or nl < m.end()):
                cut = nl + 1
            elif m is not None:
                cut = m.end()
            elif len(buf) > max_chars:
                # very long run-on: break at the last comma or space
                cut = max(buf.rfind(", ", 0, max_chars), buf.rfind(" ", 0, max_chars)) + 1
                if cut <= 0:
                    cut = max_chars
            if cut is None:
                break
            chunk, buf = clean_for_speech(buf[:cut]), buf[cut:]
            if _ONLY_TAGS.fullmatch(chunk):
                carry = f"{carry} {chunk}".strip()
            elif chunk:
                first = False
                yield f"{carry} {chunk}".strip()
                carry = ""
    tail = f"{carry} {clean_for_speech(buf)}".strip()
    if tail and not _ONLY_TAGS.fullmatch(tail):  # a lone closing <chuckle> = seconds of silence, then a laugh
        yield tail
