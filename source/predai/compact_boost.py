"""Dense text+identity booster used as an optional cold-start branch.

Benchmark IDs are never features. Unseen model names fall back through
family tokens, smoothed provider/harness rates, and hashed item text.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.isotonic import IsotonicRegression

from predai.ensemble_v2 import features as legacy_features
from predai.features import _s, detect_domain, item_handcrafted

FAMILIES = (
    "llama", "qwen", "gpt", "claude", "deepseek", "gemini", "mistral", "mixtral",
    "command", "phi", "yi", "wizard", "secalign", "unifiedqa", "t5", "gemma",
    "grok", "kimi", "glm", "internlm",
)
VARIANTS = (
    "instruct", "chat", "base", "mini", "nano", "pro", "flash", "sonnet", "opus",
    "thinking", "coder", "reason", "turbo", "haiku",
)
_CTX = re.compile(r"([\d.]+)\s*([km]?)", re.I)
_YEAR = re.compile(r"\b(20\d\d)(?:-(\d\d))?")


def _date_num(value: str) -> float:
    match = _YEAR.search(_s(value))
    if not match:
        return 0.0
    return (int(match.group(1)) - 2020) + (int(match.group(2) or 1) - 1) / 12.0


def _context_log(item: dict) -> float:
    text = _s(item.get("item_features"))
    attrs = dict(part.split("=", 1) for part in text.split(";") if "=" in part)
    match = _CTX.match(attrs.get("context_length", ""))
    if not match:
        return 0.0
    scale = {"": 1.0, "k": 1024.0, "m": 1048576.0}[match.group(2).lower()]
    return float(math.log1p(float(match.group(1)) * scale))


def numeric_row(pair: list) -> np.ndarray:
    subject, item = pair
    name = _s(subject.get("normalized_name") or subject.get("display_name")).lower()
    effort = _s(subject.get("reasoning_effort")).lower()
    content = _s(item.get("item_content"))[:12000]
    hand = item_handcrafted(item)
    family = [float(token in name) for token in FAMILIES]
    variant = [float(token in name) for token in VARIANTS]
    size = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name)
    domain = detect_domain(item)
    row = [
        *hand,
        *family,
        *variant,
        float(any(token in effort for token in ("high", "max"))),
        float(any(token in effort for token in ("low", "min", "none", "minimal"))),
        float(sum("\u4e00" <= ch <= "\u9fff" for ch in content) / max(len(content), 1)),
        float(min(10, len(re.findall(r"(?:^|\n)\s*[A-E][.):]", content)))),
        _context_log(item),
        _date_num(subject.get("release_date")),
        _date_num(subject.get("access_date")),
        float(math.log1p(float(size.group(1)))) if size else 0.0,
        float(size is not None),
        float(bool(effort)),
        math.log1p(len(name)) / 5.0,
        float(domain == "code"),
        float(domain == "math"),
        float(domain == "web"),
        float(domain == "research"),
        float(domain == "doc"),
    ]
    return np.asarray(row, dtype=np.float32)


def _smooth_rate(labels: np.ndarray, keys: list[str], prior: float, strength: float = 20.0) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for key, label in zip(keys, labels):
        buckets.setdefault(key, []).append(float(label))
    return {
        key: (sum(values) + strength * prior) / (len(values) + strength)
        for key, values in buckets.items()
    }


class CompactBoost:
    """HistGradientBoosting on handcrafted identity features plus a text SVD."""

    def __init__(self, svd_dim: int = 48, max_iter: int = 320, leaf_nodes: int = 127):
        self.svd_dim = int(svd_dim)
        self.max_iter = int(max_iter)
        self.leaf_nodes = int(leaf_nodes)

    def fit(self, pairs, labels, sample_weight):
        labels = np.asarray(labels, dtype=np.float64)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        order = np.arange(len(labels))
        rng = np.random.default_rng(2026)
        rng.shuffle(order)
        cut = max(1, int(len(order) * 0.9))
        train_idx, calib_idx = order[:cut], order[cut:]
        train_pairs = [pairs[i] for i in train_idx]
        self._fit_rates(train_pairs, labels[train_idx])
        self.word_hasher = HashingVectorizer(n_features=2**15, ngram_range=(1, 2), dtype=np.float32)
        self.word_svd = TruncatedSVD(n_components=self.svd_dim, n_iter=4, random_state=2026)
        texts = [legacy_features(pair)[1] for pair in train_pairs]
        word = self.word_svd.fit_transform(self.word_hasher.transform(texts)).astype(np.float32)
        numeric = self._numeric_matrix(train_pairs)
        self.tree = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=self.max_iter,
            max_leaf_nodes=self.leaf_nodes,
            min_samples_leaf=25,
            l2_regularization=0.2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=2026,
        )
        self.tree.fit(np.hstack([numeric, word]), labels[train_idx], sample_weight=sample_weight[train_idx])
        raw = self._raw_many([pairs[i] for i in calib_idx])
        self.isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        self.isotonic.fit(raw, labels[calib_idx], sample_weight=sample_weight[calib_idx])
        return self

    def _fit_rates(self, pairs, labels) -> None:
        prior = float(np.mean(labels))
        self.global_rate = prior
        providers, harnesses, families, domains = [], [], [], []
        for pair in pairs:
            subject, item = pair
            name = _s(subject.get("normalized_name")).lower()
            matched = [token for token in FAMILIES if token in name]
            providers.append(_s(subject.get("provider")) or "unknown")
            harnesses.append(_s(subject.get("harness")) or "unknown")
            families.append(matched[0] if matched else "other")
            domains.append(detect_domain(item))
        self.provider_rate = _smooth_rate(labels, providers, prior)
        self.harness_rate = _smooth_rate(labels, harnesses, prior)
        self.family_rate = _smooth_rate(labels, families, prior)
        self.domain_rate = _smooth_rate(labels, domains, prior)

    def _numeric_matrix(self, pairs) -> np.ndarray:
        rows = []
        for pair in pairs:
            subject, item = pair
            name = _s(subject.get("normalized_name")).lower()
            matched = [token for token in FAMILIES if token in name]
            family = matched[0] if matched else "other"
            base = numeric_row(pair)
            extra = np.array([
                self.provider_rate.get(_s(subject.get("provider")) or "unknown", self.global_rate),
                self.harness_rate.get(_s(subject.get("harness")) or "unknown", self.global_rate),
                self.family_rate.get(family, self.global_rate),
                self.domain_rate.get(detect_domain(item), self.global_rate),
                self.global_rate,
            ], dtype=np.float32)
            rows.append(np.concatenate([base, extra]))
        return np.vstack(rows)

    def _raw_many(self, pairs) -> np.ndarray:
        if not pairs:
            return np.array([])
        texts = [legacy_features(pair)[1] for pair in pairs]
        word = self.word_svd.transform(self.word_hasher.transform(texts)).astype(np.float32)
        raw = self.tree.predict(np.hstack([self._numeric_matrix(pairs), word]))
        return np.clip(raw, 0.001, 0.999)

    def base_many(self, pairs, calibrated: bool | None = None) -> np.ndarray:
        raw = self._raw_many(pairs)
        if calibrated is None:
            calibrated = bool(getattr(self, "use_isotonic", True))
        if not calibrated or not len(raw):
            return raw
        return np.clip(self.isotonic.predict(raw), 0.001, 0.999)

    def base_many_raw(self, pairs) -> np.ndarray:
        return self.base_many(pairs, calibrated=False)
