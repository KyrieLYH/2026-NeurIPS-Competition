"""Build the V3 shared submission from a trained V3 final artifact.

This is the missing reproducible packaging step for
submission_v3_shared_20261014. It does not train a model; run run_v3.py
first to create the base final/ensemble.joblib.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path,
                        default=ROOT / "experiments" / "v3-rebuild" / "final")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "submission_v3_shared_rebuild")
    parser.add_argument("--shared-weight", type=float, default=0.5)
    args = parser.parse_args()

    metadata_path = args.artifacts / "training.json"
    base_path = args.artifacts / "ensemble.joblib"
    if not metadata_path.is_file() or not base_path.is_file():
        raise FileNotFoundError(
            f"Expected {base_path} and {metadata_path}; run scripts/run_v3.py --final first."
        )
    if args.out.exists() or args.out.with_suffix(".zip").exists():
        raise ValueError("Output exists; choose a new --out path to avoid overwriting a submission.")

    from predai.shared_v3 import SharedEnsemble

    base = joblib.load(base_path)
    model = SharedEnsemble(base, shared_weight=args.shared_weight)

    (args.out / "predai").mkdir(parents=True)
    (args.out / "artifacts").mkdir()
    for name in ("__init__.py", "features.py", "ensemble_v2.py",
                 "adaptive_v3.py", "shared_v3.py", "compact_boost.py"):
        shutil.copy2(ROOT / "predai" / name, args.out / "predai" / name)

    joblib.dump(model, args.out / "artifacts" / "ensemble.joblib")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["shared_adaptation"] = {
        "shared_weight": args.shared_weight,
        "task_precision": 0.5,
        "subject_precision": 2.0,
        "same_benchmark_only": True,
        "independent_fallback": True,
    }
    (args.out / "artifacts" / "training.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    (args.out / "model.py").write_text(
        '''from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
from threadpoolctl import threadpool_limits
from predai.adaptive_v3 import AdaptiveEnsemble, BoostedBase
from predai.shared_v3 import SharedEnsemble
import predai.compact_boost

_MODEL = None

def predict(input: list[dict], labeled: list | None = None) -> float:
    global _MODEL
    with threadpool_limits(limits=1):
        if _MODEL is None:
            _MODEL = joblib.load(Path(__file__).resolve().parent / "artifacts" / "ensemble.joblib")
        return float(_MODEL.predict_many([input], labeled)[0])
''', encoding="utf-8"
    )

    import numpy
    import scipy
    import sklearn
    import threadpoolctl

    versions = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    (args.out / "requirements.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in versions.items()),
        encoding="utf-8"
    )
    (args.out / "README.txt").write_text(
        "V3 ensemble with same-benchmark shared evidence adaptation.\n"
        "No labeling.py: platform random acquisition is used.\n"
        "No runtime network access or subject execution.\n",
        encoding="utf-8"
    )

    archive = args.out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for path in args.out.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                target.write(path, path.relative_to(args.out))
    print(archive)


if __name__ == "__main__":
    main()
