"""Hide personal data in captured traces before they are stored.

Masked: emails, UPI IDs, Indian PAN, Aadhaar numbers, card numbers, phone numbers,
and any other long digit run (account numbers, OTPs). Each is replaced by a typed
placeholder such as [phone], so the text keeps its shape for training.
"""

from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("upi", re.compile(r"\b[\w.-]{2,}@(?:ok)?[a-z]{2,15}\b", re.I)),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("aadhaar", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("phone", re.compile(r"(?<![\w])(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}\b|(?<![\w])\+\d[\d -]{7,14}\d\b")),
    ("number", re.compile(r"\b\d{6,}\b")),
]


def mask_text(text: str) -> str:
    if not text:
        return text
    for name, pat in PATTERNS:
        text = pat.sub(f"[{name}]", text)
    return text


def mask_obj(obj):
    """Mask every string inside a JSON-like structure (dicts, lists)."""
    if isinstance(obj, str):
        return mask_text(obj)
    if isinstance(obj, list):
        return [mask_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: mask_obj(v) for k, v in obj.items()}
    return obj
