"""Cold-start ensemble with stateless, pair-local test-time calibration.

Only public input attributes are features. Benchmark IDs are grouping keys,
never learned features. Defaults are experimental, not tuned claims.
"""
from __future__ import annotations
import json
import re
from functools import lru_cache
import numpy as np
from predai.features import item_handcrafted, subject_text, item_text, _s

BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])
SUBJECT_FIELDS = ('normalized_name', 'provider', 'release_date', 'access_date',
                  'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra')


def visible_subject(subject):
    return {k: _s(subject.get(k)) for k in SUBJECT_FIELDS}


def safe_extension(value, allowed):
    """Retain only reviewed static keys; never embed arbitrary extension blobs."""
    text = _s(value)
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = dict(part.split('=', 1) for part in text.split(';') if '=' in part)
    if not isinstance(parsed, dict):
        return ''
    return ';'.join(f'{k}={_s(v)}' for k, v in parsed.items() if k.strip() in allowed)


def features(pair):
    subject, item = pair
    # Explicit allowlist: no reference answers, traces, graders, IDs or outcomes.
    s = visible_subject(subject)
    s['subject_features_extra'] = safe_extension(s['subject_features_extra'],
        {'model_config', 'source_model_name', 'released_variant', 'source_model_variant',
         'model_identifier', 'source_model', 'source_model_identifier', 'defense',
         'input_format', 'prompt_condition', 'instruction_size', 'shot_size'})
    it = {k: _s(item.get(k)) for k in ('item_content', 'item_features', 'interactors')}
    it['item_features'] = safe_extension(it['item_features'],
        {'lang', 'website', 'paper', 'competition', 'problem_idx', 'prompt_source',
         'suite', 'context_length', 'task', 'category', 'context_condition',
         'question_polarity', 'database_id', 'difficulty', 'prompt_condition', 'capability'})
    it['item_content'] = it['item_content'][:12000]
    numeric = dict(cached_item_features(it['item_content'], it['item_features'], it['interactors']))
    for part in it['item_features'].split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            numeric[f'attribute_{k}'] = v
    match = re.search(r'(\d+(?:\.\d+)?)\s*[bB]\b', s['normalized_name'])
    numeric['model_log_size'] = float(np.log1p(float(match[1]))) if match else 0.
    numeric['model_size_known'] = float(match is not None)
    numeric.update({f'subject_{k}': v for k, v in s.items() if v})
    numeric['effort_present'] = float(bool(s['reasoning_effort']))
    return numeric, subject_text(s) + '\n' + item_text(it)[:12000]


@lru_cache(maxsize=8192)
def cached_item_features(content, attributes, interactors):
    it = {'item_content': content, 'item_features': attributes, 'interactors': interactors}
    return tuple((f'item_{j}', x) for j, x in enumerate(item_handcrafted(it)))


def pair_weights(frame):
    counts = frame.groupby(['benchmark_id', 'subject_id'])['label'].transform('size')
    w = 1.0 / counts.to_numpy()
    return w / w.mean()


class Ensemble:
    def __init__(self, kappa=8.0, weights=(.25, .375, .375), n_estimators=80, max_depth=12, min_samples_leaf=20, n_jobs=4):
        self.kappa = float(kappa)
        self.weights = np.asarray(weights, dtype=float)
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError('kappa must be finite and positive')
        if len(self.weights) != 3 or np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise ValueError('Three nonnegative ensemble weights required')
        self.weights /= self.weights.sum()

    def fit(self, pairs, labels, sample_weight):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import ExtraTreesRegressor
        from scipy.sparse import hstack
        rows = [features(p) for p in pairs]
        self.vectorizer = DictVectorizer()
        x = self.vectorizer.fit_transform([r[0] for r in rows])
        self.hasher = HashingVectorizer(n_features=2**16, ngram_range=(1, 2))
        z = hstack([x, self.hasher.transform([r[1] for r in rows])]).tocsr()
        self.linear = Ridge(alpha=100., solver='lsqr')
        self.linear.fit(z, labels, sample_weight=sample_weight)
        self.trees = ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf, n_jobs=self.n_jobs, random_state=42)
        self.trees.fit(x, labels, sample_weight=sample_weight)
        self.trees.n_jobs = 1  # Stable reduction order for repeated scoring calls.
        self.mean = float(np.average(labels, weights=sample_weight))
        return self

    def base_many(self, pairs):
        from scipy.sparse import hstack
        rows = [features(p) for p in pairs]
        x = self.vectorizer.transform([r[0] for r in rows])
        z = hstack([x, self.hasher.transform([r[1] for r in rows])]).tocsr()
        members = np.column_stack([np.full(len(pairs), self.mean),
                                   self.linear.predict(z), self.trees.predict(x)])
        return np.clip(np.clip(members, .001, .999) @ self.weights, .001, .999)

    def matching(self, pair, labeled):
        subject, item = pair
        bid = item.get('benchmark_id')
        if not bid:
            raise ValueError('Anonymous item benchmark_id is required')
        # No state shared between calls or budgets. Avoid cross-benchmark pooling.
        return [(p, float(y)) for p, y in (labeled or [])
                if p[1].get('benchmark_id') == bid and
                p[0] == subject]

    def predict_many(self, pairs, labeled=None):
        base = self.base_many(pairs)
        groups = {}
        result = []
        for pair, p0 in zip(pairs, base):
            key = json.dumps([pair[0], pair[1].get('benchmark_id')], sort_keys=True)
            if key not in groups:
                evidence = self.matching(pair, labeled) if pair[1].get('benchmark_id') else []
                if evidence:
                    ep, ys = zip(*evidence)
                    groups[key] = offset(self.base_many(ep), np.asarray(ys), self.kappa)
                else:
                    groups[key] = 0.
            logit = np.log(p0 / (1. - p0)) + groups[key]
            result.append(float(1. / (1. + np.exp(-logit))))
        return result


def offset(base, labels, kappa):
    """One-dimensional regularized logistic intercept, retaining item ranking."""
    if kappa <= 0:
        raise ValueError('kappa must be positive')
    if not np.all(np.isfinite(labels)) or np.any((labels < 0) | (labels > 1)):
        raise ValueError('Evidence must be finite in [0, 1]')
    logits = np.log(base / (1. - base))
    delta = 0.
    for _ in range(30):
        p = 1. / (1. + np.exp(-np.clip(logits + delta, -30, 30)))
        step = (np.sum(p - labels) + kappa * delta / 4.) / (np.sum(p * (1-p)) + kappa / 4.)
        delta -= float(np.clip(step, -2, 2))
        if abs(step) < 1e-8:
            break
    return delta
