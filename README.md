# Calibrated Lightweight Ensembles for Predictive AI Evaluation

Reproduction materials for the NeurIPS 2026 Predictive AI Evaluation Competition technical report.

## Selected submission

| Field | Value |
| --- | --- |
| CoDaBench submission ID | 1020550 |
| Recorded formative Brier ALC | 0.139514 |
| Original inference archive | `original_submission/submission_v10_calibrated.zip` |
| Archive size | 3,467,137 bytes |
| Archive SHA-256 | `4d1d1ada9d9e6ac74b39676e6c75647f6bc5136a299b09a62839e12e814c53af` |

The score belongs to the selected formative evaluation. The nine recorded formative evaluations use different sampled evaluation data. Their descriptive scores appear in `evidence/same_version_submissions.csv`.

## Directory layout

Run the commands below from the directory containing this README. A repository checkout may place this directory inside `V10_1020550_code_bundle/`.

| Path | Purpose |
| --- | --- |
| `original_submission/` | Original CoDaBench inference ZIP |
| `runtime/` | Extracted inference files, fitted model, and runtime metadata |
| `submission_v10_calibrated/` | Additional extracted copy of the inference files |
| `source/predai/` | Feature, model, data-loading, and adaptation implementations |
| `source/scripts/` | Historical training and experiment implementations |
| `tools/` | Reproduction, verification, and packaging entry points |
| `calibration_cache/` | Public development and confirmation predictions used for intercept fitting |
| `evidence/` | Training metadata, historical experiment summaries, and verification records |
| `requirements_runtime.txt` | Pinned inference and calibration dependencies |
| `requirements_training_reference.txt` | Pinned dependencies including training-table processing |
| `MANIFEST.json` | Historical bundle file hashes |

The inference ZIP contains the scoring runtime and fitted weights. Training code and reproduction tools are supplied in this code bundle. Distribute this directory together with the original inference ZIP so reviewers can follow the workflow below.

## Environment

The recorded environment uses Python 3.10.21. CPU execution is sufficient for inference and the supplied training recipe. In a Python 3.10 environment, install the relevant dependencies.

For inference and intercept refitting, run

```sh
python -m pip install -r requirements_runtime.txt
```

For base training and historical experiment imports, run

```sh
python -m pip install -r requirements_training_reference.txt
```

| Package | Recorded version |
| --- | --- |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |
| pandas | 2.3.3 |
| pyarrow | 25.0.1 |

Inference loads local fitted weights and performs local feature construction and adaptation. Installation and public-data retrieval require their usual network access. Full training reads a candidate pool of 55,047,908 response rows into memory before sampling. Provision memory for that pool and its intermediate tables. A measured peak-memory requirement is unavailable.

## 1. Check bundle integrity

```sh
python tools/verify_bundle.py
```

The verifier checks every entry in `MANIFEST.json`, then compares `runtime/` files against the original ZIP members. Success prints a JSON object with `status` equal to `PASS` and `runtime_members_identical` equal to `true`.

The historical manifest also lists `report_reference/` materials and `START_HERE_给新GPT.md`. This checkout currently lacks those entries, so the full manifest check reports them as missing. Resolve those manifest entries before distributing a verified release. Changes to this README also require its manifest hash to match the distributed file.

## 2. Run original-weight inference checks

```sh
python tools/check_submission.py
```

This command extracts the original ZIP into a temporary directory and loads it in an isolated Python subprocess. It checks all six budgets, finite native-float outputs, determinism, input preservation, benchmark isolation, long-text handling, JSON attributes, label adaptation, and missing fields. Success prints a JSON object with `status` equal to `PASS`.

For a minimal direct call, run

```sh
python -c "import sys; sys.path.insert(0, 'runtime'); from model import predict; subject = dict.fromkeys(('normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra'), ''); subject['normalized_name'] = 'example-7B'; item = {'benchmark_id':'example','item_content':'An example question','item_features':'','interactors':''}; print(predict([subject, item], []))"
```

The interface is `predict(input, labeled)`, where `input` is `[subject, item]`. Each revealed-label entry is `[[subject, item], outcome]`, with outcome equal to zero or one. At budget zero, pass an empty list. The platform supplies the evidence allowed at each budget. The package uses default random acquisition.

## 3. Refit the selected calibration intercept

```sh
python tools/fit_selected_calibration.py
```

The script reads both prediction caches in `calibration_cache/`, fits an intercept with slope fixed at one, and checks its agreement with the exported metadata to absolute tolerance `1e-10`.

Expected values are

| Quantity | Value |
| --- | --- |
| Public prediction episodes | 758 |
| Public benchmarks | 47 |
| Slope | 1.0 |
| Intercept | -0.5796672067792444 |
| Penalized fitting objective | 0.23375083370787164 |

An optional result file can be written to a new path.

```sh
python tools/fit_selected_calibration.py --output outputs/calibration_refit.json
```

These caches contain public-task predictions and outcomes. The final intercept fit includes development and confirmation episodes. Its objective measures fit to those cached outcomes. Independent confirmation results in the report use the earlier frozen intercept described in the next section.

## 4. Recreate the final base training recipe

### Public data preparation

The training source is the public Measurement Data Bank. Obtain the public release tables and retain benchmark subdirectories under a common data root. The loader accepts the following layout, including the singular filename `response.parquet` as an alternative to `responses.parquet`.

```text
measurement-db/
  benchmark_name/
    formatted_tables/
      subjects.parquet
      items.parquet
      responses.parquet
      benchmarks.parquet
```

Tables may also be placed directly under each benchmark directory. The catalog `benchmarks.parquet` is optional for legacy layouts. When present, its release status is checked. Table columns must follow the public measurement-db schema expected by `source/scripts/finetune_current.py`.

The loader excludes `bigfinancebench`, non-public releases, and benchmarks containing invalid or nonbinary outcomes. It checks entity joins and response identifiers. The training entry requires the retained set of 105 benchmarks in `evidence/base_training_metadata.json` and exactly 55,047,908 candidate responses.

The public dataset is available from [Measurement Data Bank](https://huggingface.co/datasets/aims-foundations/measurement-db).

### Training command

Replace the data-root argument with the local directory containing the benchmark folders. Use a new output directory and a new sample-output path.

```sh
python tools/train_selected_base.py --raw-data /path/to/measurement-db --output-dir outputs/retrained_v10 --sample-out outputs/final_sample_150k.parquet --threads 4
```

The command samples 150,000 rows using seed zero, attaches public input features, computes subject-benchmark-balanced weights, and fits the constant/Ridge/ExtraTrees ensemble. It wraps the fitted base with the released global intercept and adaptation configuration. Outputs include

- `sample_ids.parquet`, containing sampled identifiers.
- `base.joblib`, containing the fitted base ensemble.
- `ensemble.joblib`, containing the calibrated predictor.
- `training.json`, containing parameters and training metadata.

A saved feature-attached final sample can be supplied for subsequent runs.

```sh
python tools/train_selected_base.py --sample-in outputs/final_sample_150k.parquet --output-dir outputs/retrained_v10_replay --threads 4
```

The original public-data revision and original final sample manifest are unavailable. Exact historical weight reconstruction therefore remains limited by snapshot availability, row ordering, sampling, and environment. The bundled weights support original-weight inference. The training entry supports recreation of the fixed recipe with compatible public tables. It reuses the released intercept.

## 5. Interpret and inspect historical experiments

| Stage | Base and calibration setting |
| --- | --- |
| Development | 123,060-row base excluding 47 benchmarks, 698 episodes across 41 benchmarks, benchmark-excluded calibration assignments |
| Confirmation | Same validation base, 60 episodes across six benchmarks, frozen intercept approximately -0.4100424649 |
| Final calibration | Intercept refit on all 758 cached episodes across 47 public benchmarks |
| Exported submission | Original 150,000-row full-public base with final intercept approximately -0.5796672068 |

Use `evidence/selection_historical.json` and `evidence/confirmation_historical.json` to inspect historical summaries. `evidence/README.md` explains the scope of each evidence file. `evidence/handoff_validation.json` records prior checks performed in the historical environment.

The original experiment pipeline is implemented in `source/scripts/run_cal_v10.py`. It references earlier samples, episode pools, and fitted validation models under `source/experiments/`, together with raw data under `source/datasets/raw/`. Those historical inputs are absent from this distribution. Full candidate-search replay requires restoring them. The two bundled caches support the final intercept refit directly through Step 3.

## 6. Package the runtime

```sh
python tools/package_submission.py
python tools/check_submission.py outputs/submission_v10_repacked.zip
```

The default packager writes a new archive using the original member list and the files in `runtime/`. Its output reports byte equality of members against the original archive. ZIP container metadata can produce a different archive hash while all member bytes agree. The output path must be new.

To package a newly trained model, copy `runtime/` to a separate staging directory, replace `artifacts/ensemble.joblib` and `artifacts/training.json` with the training outputs, and run

```sh
python tools/package_submission.py --runtime outputs/runtime_retrained --output outputs/submission_retrained.zip
python tools/check_submission.py outputs/submission_retrained.zip
```

The archive root contains `model.py`, `requirements.txt`, `predai/`, and `artifacts/`. The comparison reports changed members for retrained artifacts.

## Submission materials and provenance

The OpenReview supplementary materials should include this reproduction bundle and the original CoDaBench inference ZIP. The report identifies submission 1020550. Confirm that the repository URL in the report is accessible to reviewers when the materials are released.

The method uses public measurement-db tables, lexical hashing, weighted regression, global calibration, and revealed-label adaptation. OpenAI Codex assisted code implementation, experiment execution, validation, and documentation. Dataset and upstream benchmark terms are discussed in Appendix I of the technical report. The inherited V3 components are identified there. Confirm author attribution and applicable code licenses before public release.

The competition interface, evaluation metric, and submission rules are documented on the [competition website](https://aims.stanford.edu/competition).
