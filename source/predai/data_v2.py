"""Version-aware public snapshot loader with strict joins and label policies."""
from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd
import pyarrow.parquet as pq
from predai.data import list_benches, SUBJECT_KEYS
from predai.paths import RAW_DATA
from predai.features import _s
from predai.tables import table_path


def load_snapshot(root=RAW_DATA, label_policy='binary-benchmarks', benches=None,
                  max_content_chars=12000):
    if label_policy not in ('binary-benchmarks', 'soft-auxiliary'):
        raise ValueError('Unknown label policy')
    root = Path(root)
    frames, audit = [], []
    for bench in (list_benches(root) if benches is None else benches):
        info = {'benchmark': bench}
        try:
            catalog = table_path(root, bench, 'benchmarks')
        except FileNotFoundError:
            catalog = None  # Legacy snapshots may not have a catalog table.
        if catalog:
            c = pd.read_parquet(catalog)
            if 'release' in c and not c['release'].eq('public').all():
                info['excluded'] = 'not an active public release'
                audit.append(info)
                continue
        if bench == 'bigfinancebench':
            info['excluded'] = 'upstream evaluation-only / do-not-train'
            audit.append(info)
            continue
        rp = table_path(root, bench, 'responses')
        columns = [k for k in ('subject_id', 'item_id', 'response_id', 'benchmark_id',
                               'trial', 'interactors', 'response') if k in pq.read_schema(rp).names]
        r = pd.read_parquet(rp, columns=columns)
        y = pd.to_numeric(r['response'], errors='coerce')
        nonbinary = y.notna() & ~y.isin([0., 1.])
        valid = y.notna() & y.between(0, 1)
        info.update(response_rows=len(r), invalid_labels=int((~valid).sum()),
                    nonbinary_labels=int(nonbinary.sum()))
        if (y.notna() & ~y.between(0, 1)).any() or (label_policy == 'binary-benchmarks' and nonbinary.any()):
            info['excluded'] = 'fractional or non-probability response scale'
            warnings.warn(f'{bench}: excluded fractional/out-of-range response scale')
            audit.append(info)
            continue
        s = pd.read_parquet(table_path(root, bench, 'subjects'), columns=['subject_id'] + SUBJECT_KEYS)
        i = pd.read_parquet(table_path(root, bench, 'items'),
                            columns=['item_id', 'benchmark_id', 'content', 'item_features'])
        if s['subject_id'].duplicated().any() or i['item_id'].duplicated().any():
            raise ValueError(f'{bench}: duplicate entity IDs')
        if r['response_id'].duplicated().any():
            raise ValueError(f'{bench}: duplicate response IDs')
        if not r['subject_id'].isin(s['subject_id']).all() or not r['item_id'].isin(i['item_id']).all():
            raise ValueError(f'{bench}: unmatched IDs, check dataset revision')
        expected = i.set_index('item_id')['benchmark_id'].reindex(r['item_id']).to_numpy()
        if (expected != r['benchmark_id'].to_numpy()).any():
            raise ValueError(f'{bench}: item/response benchmark mismatch')
        r = r.loc[valid].copy()
        r['label'] = y.loc[valid]
        r['binary_benchmark'] = not nonbinary.any()
        # Normalize once per entity, before expanding to millions of observations.
        for col in SUBJECT_KEYS:
            s[col] = s[col].map(_s)
        i['content'] = i['content'].map(_s).str.slice(0, max_content_chars)
        i['item_features'] = i['item_features'].map(_s)
        if 'interactors' not in r:
            r['interactors'] = ''
        r['interactors'] = r['interactors'].fillna('').astype(str)
        df = r.merge(s, on='subject_id', validate='many_to_one')
        df = df.merge(i.drop(columns='benchmark_id'), on='item_id', validate='many_to_one')
        df = df.rename(columns={'content': 'item_content'})
        info.update(used_rows=len(df), subjects=len(s), items=len(i))
        audit.append(info)
        frames.append(df)
    if not frames:
        raise ValueError('No compatible training data')
    result = pd.concat(frames, ignore_index=True)
    result.attrs['audit'] = audit
    return result


def to_pair(row):
    subject = {k: _s(row[k]) for k in SUBJECT_KEYS}
    item = {k: _s(row[k]) for k in ('item_content', 'item_features', 'interactors', 'benchmark_id')}
    return [subject, item]


def balanced_sample(frame, max_rows, seed=0):
    """Equal quotas per scored pair; sample before constructing text features."""
    if not max_rows or len(frame) <= max_rows:
        return frame
    groups = frame.groupby(['benchmark_id', 'subject_id'], sort=True)
    if max_rows < groups.ngroups:
        raise ValueError('Training cap must cover at least one row per subject-benchmark pair')
    quota = max(1, max_rows // groups.ngroups)
    parts = [g.sample(min(len(g), quota), random_state=seed) for _, g in groups]
    out = pd.concat(parts)
    remaining = max_rows - len(out)
    if remaining > 0:
        rest = frame.drop(index=out.index)
        out = pd.concat([out, rest.sample(min(remaining, len(rest)), random_state=seed)])
    return out
