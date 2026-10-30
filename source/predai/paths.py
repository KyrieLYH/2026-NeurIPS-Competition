# -*- coding: utf-8 -*-
"""Project paths. Training data lives under datasets/."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"
RAW_DATA = DATASETS / "raw"
PROCESSED_DATA = DATASETS / "processed"
TRAIN_PARQUET = PROCESSED_DATA / "train.parquet"
SUBMISSION = PROJECT_ROOT / "submission"
ARTIFACTS = SUBMISSION / "artifacts"
