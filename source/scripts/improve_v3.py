"""Compare stronger training settings on the original development benchmarks.

The adaptation recipe stays inside the senior V3 model. Selection uses only
aegis, babilong, bird_sql, real_webagents, and researchcodebench. The final
model is then fit on every public binary benchmark, including those five.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predai.adaptive_v3 import adapt
from predai.data_v2 import balanced_sample, to_pair
from predai.ensemble_v2 import ALC_WEIGHTS, BUDGETS, Ensemble, pair_weights
from predai.shared_v3 import hierarchical_predict, subject_key
from scripts.finetune_current import attach_features, binary_benchmarks, load_slim
from predai.paths import RAW_DATA

DEV = ["aegis", "babilong", "bird_sql", "real_webagents", "researchcodebench"]
ADAPT = {
    "map_k2": {"boost_weight": 0.0, "method": "map", "kappa": 2.0},
    "beta2": {"boost_weight": 0.0, "method": "beta", "tau": 2.0, "retain": 0.5},
    "beta4": {"boost_weight": 0.0, "method": "beta", "tau": 4.0, "retain": 0.5},
    "beta2_keep": {"boost_weight": 0.0, "method": "beta", "tau": 2.0, "retain": 0.75},
    "bayes2": {"boost_weight": 0.0, "method": "bayes", "sd": 2.0},
}
WEIGHTS = (0.0, 0.5, 1.0)


def rng_for(seed: int, benchmark: str, subject: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{benchmark}|{subject}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "little"))


def build_episodes(frame: pd.DataFrame, seed: int, cap: int) -> list[dict]:
    episodes = []
    for (benchmark, subject), group in frame.groupby(["benchmark_id", "subject_id"], sort=True):
        items = sorted(group.item_id.unique())
        if len(items) < 80:
            continue
        generator = rng_for(seed, str(benchmark), str(subject))
        generator.shuffle(items)
        acquire = items[: len(items) // 2]
        evaluate = items[len(items) // 2 :][:cap]
        evidence = []
        for item_id in acquire[:31]:
            trials = group[group.item_id == item_id]
            evidence.append(trials.iloc[int(generator.integers(len(trials)))])
        target = group[group.item_id.isin(evaluate)]
        episodes.append({
            "benchmark_id": str(benchmark),
            "subject_id": str(subject),
            "target": target,
            "evidence": pd.DataFrame(evidence),
        })
    return episodes


def materialize(episodes: list[dict]) -> list[dict]:
    rows = pd.concat([*(ep["target"] for ep in episodes), *(ep["evidence"] for ep in episodes)], ignore_index=True)
    attached = attach_features(RAW_DATA, rows.drop_duplicates(["benchmark_id", "response_id"]))
    attached["_key"] = attached["benchmark_id"].astype(str) + "|" + attached["response_id"].astype(str)
    lookup = {key: row for key, row in zip(attached["_key"], attached.to_dict("records"))}

    def pairs_for(part: pd.DataFrame) -> tuple[list, np.ndarray]:
        records = [lookup[f"{benchmark}|{response}"] for benchmark, response in zip(part.benchmark_id.astype(str), part.response_id.astype(str))]
        return [to_pair(pd.Series(record)) for record in records], np.array([float(record["label"]) for record in records])

    ready = []
    for episode in episodes:
        target_pairs, target_y = pairs_for(episode["target"])
        evidence_pairs, evidence_y = pairs_for(episode["evidence"])
        ready.append({**episode, "target_pairs": target_pairs, "target_y": target_y, "evidence_pairs": evidence_pairs, "evidence_y": evidence_y})
    return ready


def score(model: Ensemble, episodes: list[dict]) -> list[dict]:
    by_benchmark: dict[str, list[dict]] = {}
    for episode in episodes:
        by_benchmark.setdefault(episode["benchmark_id"], []).append(episode)
    rows = []
    for benchmark, group in by_benchmark.items():
        for episode in group:
            others = [other for other in group if other["subject_id"] != episode["subject_id"]]
            digest = hashlib.sha256(f"context|0|{benchmark}|{episode['subject_id']}".encode()).digest()[:8]
            companion = others[int(np.random.default_rng(int.from_bytes(digest, "little")).integers(len(others)))] if others else None
            target_base = model.base_many(episode["target_pairs"])
            evidence_base = model.base_many(episode["evidence_pairs"])
            evidence_y = episode["evidence_y"]
            other_pairs = companion["evidence_pairs"] if companion else []
            other_base = model.base_many(other_pairs) if other_pairs else np.array([])
            other_y = companion["evidence_y"] if companion else np.array([])
            target_key = subject_key(episode["target_pairs"][0][0])
            for name, config in ADAPT.items():
                curves = {weight: [] for weight in WEIGHTS}
                for budget in BUDGETS:
                    independent = adapt(config, target_base, evidence_base[:budget], evidence_y[:budget])
                    if budget and companion is not None:
                        contexts = [subject_key(pair[0]) for pair in episode["evidence_pairs"][:budget] + other_pairs[:budget]]
                        pooled = hierarchical_predict(
                            target_base,
                            np.r_[evidence_base[:budget], other_base[:budget]],
                            np.r_[evidence_y[:budget], other_y[:budget]],
                            contexts,
                            target_key,
                        )
                    else:
                        pooled = independent
                    for weight in WEIGHTS:
                        prediction = (1 - weight) * independent + weight * pooled
                        curves[weight].append(float(np.mean((prediction - episode["target_y"]) ** 2)))
                for weight, brier in curves.items():
                    rows.append({
                        "benchmark_id": benchmark,
                        "subject_id": episode["subject_id"],
                        "adapt": name,
                        "shared_weight": weight,
                        "brier": brier,
                        "alc": float(ALC_WEIGHTS @ brier),
                        "eval_responses": int(len(episode["target_y"])),
                        "context_subjects": 2 if companion is not None else 1,
                    })
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for adapt_name in ADAPT:
        for weight in WEIGHTS:
            chosen = [row for row in rows if row["adapt"] == adapt_name and row["shared_weight"] == weight]
            if not chosen:
                continue
            summary[f"{adapt_name}@w{weight:g}"] = {
                "alc": float(np.mean([row["alc"] for row in chosen])),
                "brier": np.mean([row["brier"] for row in chosen], axis=0).tolist(),
                "pairs": len(chosen),
            }
    return summary


def fit_model(frame: pd.DataFrame, rows: int, tree: dict) -> Ensemble:
    sample = balanced_sample(frame, rows, seed=0)
    sample = attach_features(RAW_DATA, sample)
    pairs = [to_pair(row) for row in sample.to_dict("records")]
    model = Ensemble(kappa=2.0, **tree)
    started = time.perf_counter()
    model.fit(pairs, sample.label.to_numpy(), pair_weights(sample))
    print(f"fit rows={len(sample)} tree={tree} seconds={time.perf_counter() - started:.1f}", flush=True)
    return model


def main() -> None:
    out = ROOT / "experiments" / "improve-v3"
    out.mkdir(parents=True, exist_ok=True)
    benches, audit = binary_benchmarks(RAW_DATA)
    missing = [name for name in DEV if name not in benches]
    if missing:
        raise SystemExit(f"development benchmarks unavailable: {missing}")
    print(f"binary={len(benches)}", flush=True)
    slim = load_slim(RAW_DATA, benches)
    dev = slim[slim.benchmark_id.isin(DEV)].copy()
    train = slim[~slim.benchmark_id.isin(DEV)].copy()
    del slim
    episodes = materialize(build_episodes(dev, seed=0, cap=80))
    print(f"dev episodes={len(episodes)} eval_responses={sum(len(ep['target_y']) for ep in episodes)}", flush=True)
    if len(episodes) < 10:
        raise SystemExit("too few development episodes for a stable choice")
    candidates = [
        ("base150", 150_000, {"n_estimators": 80, "max_depth": 12, "min_samples_leaf": 20, "n_jobs": 8}),
        ("base400", 400_000, {"n_estimators": 80, "max_depth": 12, "min_samples_leaf": 20, "n_jobs": 8}),
        ("deep400", 400_000, {"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 8, "n_jobs": 16}),
    ]
    reports = {}
    best_name = None
    best_alc = None
    for name, rows, tree in candidates:
        print(f"candidate {name}", flush=True)
        model = fit_model(train, rows, tree)
        scored = score(model, episodes)
        summary = summarize(scored)
        reports[name] = {"rows": rows, "tree": tree, "summary": summary}
        (out / f"{name}.json").write_text(json.dumps(reports[name], indent=2), encoding="utf-8")
        winner = min(summary, key=lambda key: summary[key]["alc"])
        print(f"{name} best {winner} alc={summary[winner]['alc']:.6f}", flush=True)
        if best_alc is None or summary[winner]["alc"] < best_alc:
            best_alc = summary[winner]["alc"]
            best_name = name
            best_choice = winner
        del model
    choice = reports[best_name]
    adapt_name, weight_text = best_choice.split("@w")
    selected = {
        "candidate": best_name,
        "rows": 400_000 if "400" in best_name else 150_000,
        "tree": choice["tree"],
        "adapt": ADAPT[adapt_name],
        "adapt_name": adapt_name,
        "shared_weight": float(weight_text),
        "development_alc": best_alc,
        "reports": {name: value["summary"] for name, value in reports.items()},
    }
    # The winning candidate may have been trained without the development
    # benchmarks. Refit the selected architecture on all public binary rows.
    selected["final_rows"] = max(selected["rows"], 400_000) if best_name != "base150" else 150_000
    (out / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print("SELECTED", json.dumps({k: selected[k] for k in ("candidate", "final_rows", "adapt_name", "shared_weight", "development_alc")}), flush=True)

    all_benches, _ = binary_benchmarks(RAW_DATA)
    full = load_slim(RAW_DATA, all_benches)
    final_model = fit_model(full, selected["final_rows"], selected["tree"])
    del full
    final = out / "final"
    final.mkdir(exist_ok=True)
    import joblib
    from predai.adaptive_v3 import AdaptiveEnsemble
    wrapped = AdaptiveEnsemble(final_model, None, selected["adapt"])
    joblib.dump(wrapped, final / "ensemble.joblib")
    metadata = {
        "holdout": "development benchmarks used only for selection",
        "development_benchmarks": DEV,
        "recipe": "senior_v3_selected_on_current_snapshot",
        "train_rows": selected["final_rows"],
        "config": selected["adapt"],
        "tree": selected["tree"],
        "shared_weight": selected["shared_weight"],
        "selection": selected,
        "snapshot_note": "public binary benchmarks from the 2026-10-28 core snapshot",
    }
    (final / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FINAL", final, flush=True)


if __name__ == "__main__":
    main()
