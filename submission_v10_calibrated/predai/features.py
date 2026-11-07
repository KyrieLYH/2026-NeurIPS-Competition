# -*- coding: utf-8 -*-
"""Item/subject features that are available at competition test time."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

DOMAIN_NAMES = ("code", "math", "web", "doc", "research", "other")

_RE_LATEX = re.compile(r"\\\\[a-zA-Z]+|\$\$|\\frac|\\triangle|\\mathbb")
_RE_DIFF = re.compile(r"(?i)diff --git|fail_to_pass|pass_to_pass|traceback|pull request")
_RE_CODE = re.compile(r"(?i)\b(def |class |function |impl |fn |package |#include|console\.log)")
_RE_WEB = re.compile(r"(?i)\b(click|navigate|browser|website=|login|checkout|search for)\b")
_RE_PAPER = re.compile(r"(?i)(\\documentclass|you are an expert in reproducing|here is the paper)")
_RE_URL = re.compile(r"https?://")
_RE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _s(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text.strip()


def item_text(item: dict) -> str:
    parts = [
        _s(item.get("item_content") or item.get("content")),
        _s(item.get("item_features")),
        _s(item.get("interactors")),
    ]
    return "\n".join(p for p in parts if p)


def subject_text(subject: dict) -> str:
    keys = (
        "normalized_name",
        "display_name",
        "provider",
        "harness",
        "harness_version",
        "reasoning_effort",
        "subject_features_extra",
        "release_date",
    )
    return " | ".join(_s(subject.get(k)) for k in keys)


def detect_domain(item: dict) -> str:
    feats = _s(item.get("item_features")).lower()
    text = item_text(item)
    head = text[:4000]

    if re.search(r"(?:^|;)lang=(?:python|c\+\+|c|java|javascript|typescript|go|rust|ruby)(?:;|$)", feats):
        return "code"
    if feats.startswith("website=") or "website=" in feats:
        return "web"
    if feats.startswith("paper=") or "paper=" in feats:
        return "research"
    if _RE_PAPER.search(head):
        return "research"
    if _RE_DIFF.search(head) or _RE_CODE.search(head[:2000]):
        return "code"
    if _RE_LATEX.search(head) or "prove that" in head.lower():
        return "math"
    if _RE_WEB.search(head):
        return "web"
    if "citation" in head.lower() or "according to the" in head.lower():
        return "doc"
    if len(text) < 500 and "?" in head:
        return "doc"
    return "other"


def length_bin(n: int) -> str:
    if n < 80:
        return "xs"
    if n < 300:
        return "s"
    if n < 1200:
        return "m"
    if n < 5000:
        return "l"
    if n < 20000:
        return "xl"
    return "xxl"


def item_handcrafted(item: dict) -> list[float]:
    text = item_text(item)
    n = max(len(text), 1)
    head = text[:8000]
    letters = sum(ch.isalpha() for ch in head)
    digits = sum(ch.isdigit() for ch in head)
    space = sum(ch.isspace() for ch in head)
    domain = detect_domain(item)
    feats = [
        math.log1p(n) / 12.0,
        min(n / 20000.0, 1.0),
        digits / max(len(head), 1),
        space / max(len(head), 1),
        letters / max(len(head), 1),
        float(bool(_RE_LATEX.search(head))),
        float(bool(_RE_DIFF.search(head))),
        float(bool(_RE_CODE.search(head))),
        float(bool(_RE_WEB.search(head))),
        float(bool(_RE_PAPER.search(head))),
        float(bool(_RE_URL.search(head))),
        float("```" in head or "diff --git" in head),
        float(head.count("\n") / 200.0),
        float(min(head.count("{"), 50) / 50.0),
        float(DOMAIN_NAMES.index(domain) / 5.0),
        float(["xs", "s", "m", "l", "xl", "xxl"].index(length_bin(n)) / 5.0),
    ]
    feats.extend(float(domain == name) for name in DOMAIN_NAMES)
    return feats


def hash_vector(text: str, dim: int = 512, prefix: str = "") -> list[float]:
    """Signed feature hashing of word n-grams. Stateless and sklearn-free."""
    vec = [0.0] * dim
    blob = (prefix + " " + (text or "")).lower()[:4000]
    tokens = _RE_IDENT.findall(blob)[:80]
    grams: list[str] = list(tokens)
    grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    compact = re.sub(r"\s+", " ", blob)[:240]
    grams.extend(compact[i : i + 5] for i in range(0, max(len(compact) - 4, 0), 6))
    for g in grams:
        digest = hashlib.md5(g.encode("utf-8", "ignore")).digest()
        h = int.from_bytes(digest[:8], "little")
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def item_vector(item: dict, dim: int = 512) -> list[float]:
    return hash_vector(item_text(item), dim=dim, prefix="item") + item_handcrafted(item)


def subject_handcrafted(subject: dict, provider_vocab: list[str], harness_vocab: list[str]) -> list[float]:
    provider = _s(subject.get("provider"))
    harness = _s(subject.get("harness"))
    effort = _s(subject.get("reasoning_effort")).lower()
    name = _s(subject.get("normalized_name") or subject.get("display_name"))
    feats = [
        float(bool(provider)),
        float(bool(harness)),
        float("think" in name.lower() or "thinking" in name.lower()),
        float(any(k in effort for k in ("high", "max", "think"))),
        float(any(k in effort for k in ("low", "min", "none"))),
        math.log1p(len(name)) / 5.0,
    ]
    feats.extend(float(provider == p) for p in provider_vocab)
    feats.extend(float(harness == h) for h in harness_vocab)
    return feats


def subject_vector(
    subject: dict,
    dim: int = 256,
    provider_vocab: list[str] | None = None,
    harness_vocab: list[str] | None = None,
) -> list[float]:
    provider_vocab = provider_vocab or []
    harness_vocab = harness_vocab or []
    return hash_vector(subject_text(subject), dim=dim, prefix="subj") + subject_handcrafted(
        subject, provider_vocab, harness_vocab
    )


def same_subject(a: dict, b: dict) -> bool:
    keys = (
        "normalized_name",
        "provider",
        "harness",
        "harness_version",
        "reasoning_effort",
        "subject_features_extra",
        "release_date",
        "access_date",
    )
    return all(_s(a.get(k)) == _s(b.get(k)) for k in keys)


def same_item(a: dict, b: dict) -> bool:
    return item_text(a) == item_text(b)


def clip_prob(p: float, eps: float = 0.02) -> float:
    if p != p:  # NaN
        return 0.5
    return float(min(1.0 - eps, max(eps, p)))
