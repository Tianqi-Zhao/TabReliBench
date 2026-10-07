# TabReliBench

Code for *TabReliBench: Evaluating Tabular Foundation Models Beyond Predictive Accuracy*.

This repository contains benchmark evaluation, metric computation, training-data meta-features, meta-predictor fitting, association analyses, and conformal prediction. Analysis commands produce numerical tables; paper plotting code is not included.

## Benchmark

| Task           | OpenML dataset list                             | TFMs | Conventional configurations |
| -------------- | ----------------------------------------------- | ---- | --------------------------- |
| Classification | [160 datasets](dataset_ids_classification.list) | 11   | 5                           |
| Regression     | [81 datasets](dataset_ids_regression.txt)       | 9    | 5                           |

All experiments use seeds 0–4, ratio 1, at most 10,000 rows, and the shared 50/50 train/test split. Classification uses stratified sampling. The main TFM panel uses eight ensemble members; conventional configurations run once per seed. The ensemble study uses E ∈ {1, 4, 8, 16, 32}. Regression interval metrics use α ∈ {0.05, 0.10, 0.15, 0.20}. LimiX-2 and TabFM regression provide R² only.

## Metrics

| Property | Classification | Regression |
| --- | --- | --- |
| Point prediction | Accuracy | R² |
| Proper score | Brier score | CRPS |
| Marginal calibration | Confidence ECE | PIT–KS, marginal coverage error |
| Classwise / subgroup calibration | Classwise ECE | Worst-slab coverage (WSC) error |
| Interval efficiency | — | Normalized interval width |

Accuracy and R² measure point-prediction performance; higher is better. Brier score and CRPS assess predictive distributions; lower is better. Confidence ECE measures how closely confidence matches accuracy, while classwise ECE checks calibration for each class. PIT–KS measures how far regression predictive quantiles depart from calibration. Marginal and WSC coverage errors measure deviations from nominal coverage overall and in a selected poorly covered feature subgroup. Lower calibration errors are better; narrower normalized intervals are preferable at comparable coverage.

## Installation

Use Python 3.12 and install a compatible PyTorch build, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export TABPFN_CHECKPOINT_DIR=/absolute/path/to/checkpoints
```

`requirements.txt` covers TabPFNv2/v2.5/v3, TabICLv1/v2, TabDPT1.1, RealMLP and XGBoost. Install other TFMs separately using the pinned sources below; keep data-processing versions consistent. BART additionally requires `pip install 'bartz[cuda12]>=0.12,<0.13'` for CUDA 12. OpenML data are downloaded automatically; pretrained weights must be obtained separately.

All foundation models: versions and weights

Software runs the model; checkpoints contain its pretrained weights. Paths below are relative to `TABPFN_CHECKPOINT_DIR`; choose the task-specific filename where alternatives appear in braces. C = classification, R = distributional regression, P = point-only regression.

| Paper model / CLI ID      | Tasks | Inference software                                                                                                                | Checkpoint path                                                                                                                            |
| ------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| TabPFNv2 / `tabpfnv2`     | C, R  | `tabpfn==8.0.2`                                                                                                                   | `tabpfn-v2-{classifier,regressor}-v2_default.ckpt`                                                                                         |
| TabPFNv2.5 / `tabpfnv2.5` | C, R  | `tabpfn==8.0.2`                                                                                                                   | `tabpfn-v2.5-{classifier,regressor}-v2.5_default.ckpt`                                                                                     |
| TabPFNv3 / `tabpfnv3`     | C, R  | `tabpfn==8.0.2`                                                                                                                   | `tabpfn-v3-{classifier,regressor}-v3_default.ckpt`                                                                                         |
| TabPFNv3.5 / `tabpfnv3.5` | C, R  | `tabpfn` 9.1.0 ([pinned source](https://github.com/PriorLabs/TabPFN/tree/15f5e6b2b629b905879b9be907261416f20d0df5))               | [tabpfn-v3.5-20260909.safetensors](https://huggingface.co/Prior-Labs/tabpfn_3_5)                                                         |
| TabICLv1 / `tabicl`       | C     | `tabicl==2.1.0`                                                                                                                   | `tabicl-classifier-v1.1-20250506.ckpt`                                                                                                     |
| TabICLv2 / `tabiclv2`     | C, R  | `tabicl==2.1.0`                                                                                                                   | `tabicl-{classifier,regressor}-v2-20260212.ckpt`                                                                                           |
| TabDPT1.1 / `tabdpt`      | C     | `tabdpt==1.1.13`                                                                                                                  | [tabdpt1_1.safetensors](https://huggingface.co/Layer6/TabDPT)                                                                            |
| TabDPT1.3 / `tabdpt1.3`   | C, R  | `tabdpt` 1.3.1 ([pinned source](https://github.com/layer6ai-labs/TabDPT-inference/tree/93670551adba28b186354fab9a1584c56c34aa76)) | [tabdpt1_3.safetensors](https://huggingface.co/Layer6/TabDPT)                                                                            |
| Causilo / `causilo`       | C, R  | `causilo` 1.0.3 ([pinned source](https://github.com/nums-ai/causilo/tree/4d26d497de28734db52c6bfc2ea949c12cc17308))               | [causilo/{classifier,regressor}/model.safetensors](https://huggingface.co/nums-ai/causilo/tree/94f2bd91db0737d4da59f347910662905ecb5a09) |
| LimiX-2 / `limix2`        | C, P  | [Pinned LimiX source](https://github.com/limix-ldm-ai/LimiX/tree/516bf396333feb3198cf7aff8a6c10421f218e24)                        | [LimiX-2.ckpt](https://huggingface.co/stable-ai/LimiX-2)                                                                                 |
| TabFM / `tabfm`           | C, P  | `tabfm[pytorch]` 1.0.1 ([pinned source](https://github.com/google-research/tabfm/tree/fbb665569425fd2f490c6576b3af967876fe11ff))  | [tabfm/{classification,regression}/model.safetensors](https://huggingface.co/google/tabfm-1.0.0-pytorch)                                 |

For source-pinned packages, use a separate environment and install with `pip install 'PACKAGE @ git+REPOSITORY_URL@COMMIT'`, using the linked revision (`tabfm[pytorch]` for TabFM). For LimiX-2, clone the pinned source, install its `environment.yml`, and add the source root to `PYTHONPATH`; do not install the unrelated PyPI `limix` package.

Causilo and TabFM require the official `config.json` beside each weight file. LimiX-2 classification requires 2–10 context classes. Conventional baselines are trained per split and need no pretrained weights.

LimiX uses the bundled [classification](evaluation/adapters/configs/limix2_classification.json) and [regression](evaluation/adapters/configs/limix2_regression.json) configurations with their [upstream license](evaluation/adapters/configs/LIMIX_LICENSE.txt). Member selection covers preprocessing variants independently of scores; requests beyond the native list repeat configurations with member-specific seeds.

## 1. Evaluate models and compute metrics

Run from the repository root. Select only models installed in the active environment:

```bash
python evaluate_vanilla_tabpfn.py --dataset_id 11 --task classification \
  --models tabpfnv3 --seed 0 --ratio 1 --n_estimators 8 --output_dir outputs/ne8
python evaluate_vanilla_tabpfn.py --dataset_id 1097 --task regression \
  --models tabpfnv3 --seed 0 --ratio 1 --n_estimators 8 --output_dir outputs/ne8
python evaluate_baselines.py --dataset_id 1097 --task regression \
  --models realmlp realmlp_hpo xgboost_quantile xgboost_quantile_hpo bart \
  --seed 0 --ratio 1 --output_dir outputs/ne8
python compute_metrics.py --results_dir outputs/ne8
```

For the full benchmark, repeat these commands for each dataset, seeds 0–4, and each applicable model. Outputs are stored under `<output>/<task>/{predictions,metrics}`. Use a separate output root for each ensemble size.

## 2. Extract meta-features and export analysis tables

```bash
python extract_datasets_meta_features.py --task classification \
  --dataset_ids_file dataset_ids_classification.list \
  --seeds 0 1 2 3 4 --features_cache_dir features_cache/classification
python extract_datasets_meta_features.py --task regression \
  --dataset_ids_file dataset_ids_regression.txt \
  --seeds 0 1 2 3 4 --features_cache_dir features_cache/regression
python -m paper_analysis.training_scales --output outputs/training_scales.tsv
python -m paper_analysis.tables --results outputs/ne8 --features features_cache \
  --training-scales outputs/training_scales.tsv --ensemble 8 --output outputs/tables
```

Meta-features use training data only. Normalized CRPS divides each trial's CRPS by the population standard deviation of that trial's training response, before averaging seeds. Exported tables retain missing/nonfinite values and seed counts; no missing result is imputed. A full paper run requires both tasks and all five seeds.

## 3. Run numerical analyses

```bash
python -m paper_analysis.benchmark --tables outputs/tables --output outputs/benchmark
python -m paper_analysis.components --tables outputs/tables --output outputs/meta
python -m paper_analysis.associations --tables outputs/tables --output outputs/associations
```

- `benchmark`: dataset-equal ranks and bootstrap intervals, score decomposition, split stability, and absolute score summaries.
- `components`: shared and model-specific score prediction using ridge and Extra-Trees; five repeats of five-fold dataset-held-out CV, with preprocessing and ridge tuning fitted inside the training folds.
- `associations`: Spearman correlations for across-model median and individual scores, BH correction over eligible candidates, and per-metric top-five unions.

For the ensemble study, export each execution root with `paper_analysis.tables`, its matching `--ensemble`, and `--training-scales outputs/training_scales.tsv`, then pass all five table directories:

```bash
python -m paper_analysis.ensemble \
  --tables outputs/tables_ne1 outputs/tables_ne4 outputs/tables_ne8 \
    outputs/tables_ne16 outputs/tables_ne32 --output outputs/ensemble
```

## 4. Conformal prediction

Reuse the saved regression predictions for the seven distributional TFMs and BART. The original held-out rows are split into auxiliary A (25%), calibration C (25%), and test T (50%). SCP and CalLCP/RLCP calibrate on A∪C; SLCP, PCP and RCP fit on A and calibrate on C. RCP fits its score model on A and calibrates on the full C, using the corresponding model package and checkpoint.

```bash
python -m conformal prepare-predictions \
  --input outputs/ne8/regression/predictions/dataset_1097_seed0_ratio1.0_tabpfnv3.pkl \
  --n-estimators 8 --output outputs/conformal/inputs/dataset_1097_seed0_tabpfnv3.pkl
python -m conformal run --input outputs/conformal/inputs --output outputs/conformal/run \
  --methods Vanilla SCP-ABS SCP-NORM SCP-CQR-standard SCP-PIT \
    CalLCP-standard RLCP-standard SLCP-standard PCP RCP-standard \
  --alphas 0.05 0.1 0.15 0.2
python -m paper_analysis.export_conformal --results outputs/conformal/run \
  --output outputs/conformal/scalars.jsonl.gz
python -m paper_analysis.conformal --input outputs/conformal/scalars.jsonl.gz \
  --output outputs/conformal/analysis
```

Prepare one input per dataset, seed and base model; for BART, omit `--n-estimators`. Use the explicit method list above: `-standard` selects signed CQR scores, and `Vanilla` provides the paired uncorrected intervals. PCP uses the [official implementation](conformal/_vendor/pcp/README.md).

The analysis commands expect the full paper panel: 81 datasets, five seeds and eight base models. They report missing configurations, use a common valid panel, and compute dataset-equal summaries with paired bootstrap intervals. For smaller runs, use `python -m conformal summarize --input outputs/conformal/run/regression/metrics --output outputs/conformal/summary`. Run any command with `--help` for all options.
