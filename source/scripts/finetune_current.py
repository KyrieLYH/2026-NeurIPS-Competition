"""Retrain the frozen V3 recipe on the current public binary snapshot.

Hyperparameters stay at the 2026-10-14 development choice, w0_beta2:
boost_weight=0, method=beta, tau=2, retain=0.5. Rows are sampled with the
original balanced_sample rule, but item text is attached only after sampling.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predai.adaptive_v3 import AdaptiveEnsemble
from predai.data import SUBJECT_KEYS, list_benches
from predai.data_v2 import balanced_sample, to_pair
from predai.ensemble_v2 import Ensemble, pair_weights
from predai.features import _s
from predai.paths import RAW_DATA
from predai.tables import table_path

CONFIG_NAME = "w0_beta2"
CONFIG = {"boost_weight": 0.0, "method": "beta", "tau": 2.0, "retain": 0.5}
FINAL_ROWS = 150_000


def binary_benchmarks(root: Path) -> tuple[list[str], list[dict]]:
    import pyarrow as pa
    import pyarrow.compute as pc

    audit = []
    kept = []
    for bench in list_benches(root):
        info = {"benchmark": bench}
        if bench == "bigfinancebench":
            info["excluded"] = "upstream evaluation-only / do-not-train"
            audit.append(info)
            continue
        try:
            catalog = table_path(root, bench, "benchmarks")
        except FileNotFoundError:
            catalog = None
        if catalog:
            catalog_frame = pd.read_parquet(catalog)
            if "release" in catalog_frame and not catalog_frame["release"].eq("public").all():
                info["excluded"] = "not an active public release"
                audit.append(info)
                continue
        path = table_path(root, bench, "responses")
        nonbinary = invalid = rows = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["response"], batch_size=1_000_000):
            num = pc.cast(batch.column(0), pa.float64(), safe=False)
            finite = pc.and_(pc.is_valid(num), pc.invert(pc.is_nan(num)))
            in_range = pc.and_(pc.greater_equal(num, 0), pc.less_equal(num, 1))
            valid = pc.and_(finite, in_range)
            invalid += pc.sum(pc.invert(pc.fill_null(valid, False))).as_py()
            is_binary = pc.or_(pc.equal(num, 0), pc.equal(num, 1))
            bad = pc.and_(pc.fill_null(valid, False), pc.invert(pc.fill_null(is_binary, False)))
            nonbinary += pc.sum(bad).as_py()
            rows += len(batch)
        info.update(response_rows=rows, invalid_labels=int(invalid), nonbinary_labels=int(nonbinary))
        if invalid or nonbinary:
            info["excluded"] = "fractional or non-probability response scale"
            audit.append(info)
            continue
        kept.append(bench)
        audit.append(info)
    return kept, audit


def load_slim(root: Path, benches: list[str]) -> pd.DataFrame:
    frames = []
    for bench in benches:
        response_path = table_path(root, bench, "responses")
        schema = set(pq.read_schema(response_path).names)
        columns = [name for name in ("subject_id", "item_id", "response_id", "benchmark_id", "trial", "response") if name in schema]
        responses = pd.read_parquet(response_path, columns=columns)
        subjects = pd.read_parquet(table_path(root, bench, "subjects"), columns=["subject_id"])
        items = pd.read_parquet(table_path(root, bench, "items"), columns=["item_id", "benchmark_id"])
        if subjects["subject_id"].duplicated().any() or items["item_id"].duplicated().any():
            raise ValueError(f"{bench}: duplicate entity IDs")
        if responses["response_id"].duplicated().any():
            raise ValueError(f"{bench}: duplicate response IDs")
        if not responses["subject_id"].isin(subjects["subject_id"]).all() or not responses["item_id"].isin(items["item_id"]).all():
            raise ValueError(f"{bench}: unmatched IDs, check dataset revision")
        expected = items.set_index("item_id")["benchmark_id"].reindex(responses["item_id"]).to_numpy()
        if (expected != responses["benchmark_id"].to_numpy()).any():
            raise ValueError(f"{bench}: item/response benchmark mismatch")
        responses["label"] = pd.to_numeric(responses["response"], errors="coerce")
        frames.append(responses[["benchmark_id", "subject_id", "item_id", "response_id", "trial", "label"]])
        print(f"slim {bench} rows={len(responses)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def attach_features(root: Path, sample: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for bench, group in sample.groupby("benchmark_id", sort=False):
        subjects = pd.read_parquet(table_path(root, bench, "subjects"))
        keep = ["subject_id"] + [column for column in SUBJECT_KEYS if column in subjects.columns]
        subjects = subjects[keep]
        for column in SUBJECT_KEYS:
            if column not in subjects:
                subjects[column] = ""
            subjects[column] = subjects[column].map(_s)
        items = pd.read_parquet(
            table_path(root, bench, "items"),
            columns=["item_id", "content", "item_features"],
        )
        items = items[items["item_id"].isin(set(group["item_id"]))]
        items["content"] = items["content"].map(_s).str.slice(0, 12000)
        items["item_features"] = items["item_features"].map(_s)
        response_path = table_path(root, bench, "responses")
        names = set(pq.read_schema(response_path).names)
        if "interactors" in names:
            interactions = pd.read_parquet(response_path, columns=["response_id", "interactors"])
            interactions = interactions[interactions["response_id"].isin(set(group["response_id"]))]
            interactions["interactors"] = interactions["interactors"].fillna("").astype(str)
        else:
            interactions = pd.DataFrame({"response_id": group["response_id"].unique(), "interactors": ""})
        merged = group.merge(subjects, on="subject_id", validate="many_to_one")
        merged = merged.merge(items, on="item_id", validate="many_to_one")
        merged = merged.merge(interactions, on="response_id", validate="many_to_one")
        parts.append(merged.rename(columns={"content": "item_content"}))
        print(f"attached {bench} rows={len(merged)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    out = ROOT / "experiments" / "finetune-current"
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    print("data", RAW_DATA, flush=True)
    started = time.perf_counter()
    benches, audit = binary_benchmarks(RAW_DATA)
    print(f"binary benchmarks={len(benches)} seconds={time.perf_counter() - started:.1f}", flush=True)
    slim = load_slim(RAW_DATA, benches)
    counts = slim.groupby("benchmark_id").size().to_dict()
    for row in audit:
        if row.get("excluded") is None:
            row["used_candidate_rows"] = int(counts.get(row["benchmark"], 0))
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"candidate rows={len(slim)}", flush=True)
    train = balanced_sample(slim, FINAL_ROWS, seed=0)
    del slim
    train = attach_features(RAW_DATA, train)
    pairs = [to_pair(row) for row in train.to_dict("records")]
    weights = pair_weights(train)
    print(
        f"fit rows={len(train)} benchmarks={train.benchmark_id.nunique()} "
        f"subjects={train.subject_id.nunique()}",
        flush=True,
    )
    fit_started = time.perf_counter()
    legacy = Ensemble(kappa=2.0).fit(pairs, train.label.to_numpy(), weights)
    model = AdaptiveEnsemble(legacy, None, CONFIG)
    print(f"fit seconds={time.perf_counter() - fit_started:.1f}", flush=True)
    import joblib
    joblib.dump(model, final / "ensemble.joblib")
    metadata = {
        "holdout": "none",
        "recipe": "senior_v3_frozen_w0_beta2",
        "training_benchmarks": sorted(train.benchmark_id.unique()),
        "train_rows": int(len(train)),
        "config": CONFIG,
        "selection": CONFIG_NAME,
        "selection_source": "submission_v3_shared_20261014 development selection; not retuned",
        "final_rows_cap": FINAL_ROWS,
        "snapshot_audit": audit,
    }
    (final / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FINAL", final, flush=True)


if __name__ == "__main__":
    main()
