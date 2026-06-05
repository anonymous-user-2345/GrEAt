# GrEAt: Generalizable Energy-Based Anomaly Detection for Tabular EHR Data

This repository contains an anonymous implementation of **GrEAt**, a fully energy-based anomaly detection framework for mixed numerical and categorical tabular data. The method is designed for tabular EHR and biomedical datasets, where anomalies may arise from feature-level deviations, multi-feature semantic inconsistencies, or distributional shifts.

GrEAt learns a scalar energy score for each record. Lower energy indicates compatibility with the learned normal-data manifold, while higher energy indicates anomalous or semantically inconsistent records.

## Repository contents

```text
GrEAt.ipynb        Main notebook for training, evaluation, mutation analysis, baselines, and ablations
README.md          Repository instructions
```

The notebook includes:

- GrEAt with an FT-Transformer encoder.
- Energy-only training with three loss components:
  - in-distribution compactness loss,
  - covariate-shift robustness loss,
  - anomaly-separation loss.
- Validation-based energy threshold selection.
- Mutation-based evaluation for unseen anomaly and robustness analysis.
- Baseline comparisons against supervised, classical OOD, neural OOD, feature-space OOD, and energy-based methods.
- Encoder ablation study.
- Objective-component ablation study.

## Method overview

GrEAt optimizes the following energy-only objective:

```text
L_total = lambda_in * L_in + lambda_cov * L_cov + lambda_anom * L_anom
```

where:

- `L_in` encourages clean normal records to have low energy.
- `L_cov` encourages benignly perturbed normal records to remain low-energy.
- `L_anom` pushes anomalous records toward high-energy regions.

During training, benign covariate-shifted samples are generated from normal records by adding small Gaussian noise to normalized numerical features. Categorical features are kept unchanged for these benign perturbations.

At test time, predictions are made by thresholding the learned energy scores. The threshold is selected on the validation split by maximizing validation F1.

## Input data format

The notebook expects a CSV file where each row is one tabular record and one column is the target label.

Example:

```text
feature_1,feature_2,feature_3,...,label
0.12,A,5.7,...,0
0.91,B,2.1,...,1
```

The label column can contain binary or multiclass labels. During preprocessing, the majority class is mapped to `0` and treated as normal, while all other classes are mapped to `1` and treated as anomalous.

To use your own dataset, edit the data-loading cell in `GrEAt.ipynb`:

```python
file_path = "your_dataset.csv"
target_col = "your_label_column"
df = pd.read_csv(file_path)
```

Optional identifier or timestamp columns can be removed before training. The notebook already includes logic for common columns such as `ID` and `Time`.

## Preprocessing

The notebook performs the following preprocessing steps:

1. Drops non-informative identifier/time columns when present.
2. Maps the majority class to normal (`0`) and minority/non-majority classes to anomalous (`1`).
3. Imputes missing numerical features using the feature mean.
4. Imputes missing categorical features using the mode.
5. Normalizes numerical features to `[0, 1]` with `MinMaxScaler`.
6. Encodes categorical features as integer IDs and passes them through learnable embeddings.
7. Uses weighted random sampling during training to reduce the effect of class imbalance.

## Main model

The main GrEAt model uses an FT-Transformer-style encoder:

- numerical features are converted into feature tokens using learnable affine transformations,
- categorical features are converted into embedding tokens,
- a learnable `[CLS]` token aggregates record-level information,
- transformer layers model feature interactions,
- a lightweight energy head maps the `[CLS]` representation to a scalar energy score.

## Evaluation settings

The notebook reports performance under four settings:

### Known anomalies

Standard held-out test evaluation using the original test labels.

### A1: feature-level mutation anomalies

Single-feature perturbations that create large feature-level deviations. These are treated as anomalies.

### A2: dependency-breaking mutation anomalies

Mutations that disrupt high-dependency feature relationships while keeping individual values plausible. These are treated as unseen semantic anomalies.

### A3: benign covariate perturbations

Small Gaussian perturbations applied to numerical features. These are used to evaluate robustness to benign noise and should not be treated as harmful semantic anomalies.

## Baselines

The notebook includes the following baseline groups:

### Supervised classifiers

- Logistic Regression
- Random Forest
- MLPClassifier
- XGBoost, if installed
- LightGBM and CatBoost hooks may be present but can be disabled or excluded from reporting

### Classical OOD detectors

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- EllipticEnvelope

### Neural OOD scoring methods

- Maximum Softmax Probability
- Maximum Logit
- Classifier Energy Score

### Feature-space OOD methods

- kNN distance
- Mahalanobis distance

### Energy-based baselines

- EnergyClassifier
- EnergyOE
- DeepSVDDEnergy

All anomaly scores are oriented so that larger values indicate more anomalous samples. Thresholds are selected on the validation split.

## Ablation studies

The notebook includes two ablation studies.

### Encoder ablation

The energy objective is kept fixed while replacing the encoder with alternative backbones:

- pretrained ALBERT,
- randomly initialized ALBERT,
- FT-Transformer,
- TabTransformer-style encoder,
- MLP encoder.

### Objective-component ablation

The FT-Transformer encoder is kept fixed while removing one objective component at a time:

- full objective,
- without `L_in`,
- without `L_cov`,
- without `L_anom`.

When one component is removed, the remaining loss weights are renormalized.

## Output files

Running the notebook can generate the following result files:

```text
great_mixed_1run_baselines_3run_great_detailed.csv
great_mixed_1run_baselines_3run_great_final.csv
great_mixed_1run_baselines_3run_great_final.xlsx
great_encoder_ablation_1run_detailed.csv
great_encoder_ablation_1run_table.csv
great_encoder_ablation_1run_table.xlsx
great_objective_ablation_1run_detailed.csv
great_objective_ablation_1run_table.csv
great_objective_ablation_1run_table.xlsx
```

The final mixed-results table reports baseline models from one run and GrEAt as the average over three random seeds.

## Installation

The notebook was developed in Python 3 with PyTorch and scikit-learn. A typical installation is:

```bash
pip install numpy pandas scikit-learn matplotlib tqdm torch transformers openpyxl
```

Optional packages for additional baselines:

```bash
pip install xgboost lightgbm catboost deepod
```

If optional packages are not installed, the corresponding baselines can be skipped.

## Running the notebook

1. Clone or download this repository.
2. Place your CSV dataset in the same directory as `GrEAt.ipynb`, or update `file_path` to the correct path.
3. Set the target column name:

```python
target_col = "label"
```

4. Run the notebook cells in order.
5. Review the generated CSV/XLSX result files.

For GPU acceleration, run the notebook in an environment with CUDA support, such as Google Colab with a GPU runtime.

## Reproducibility

The notebook sets random seeds for Python, NumPy, and PyTorch. GrEAt can be evaluated over multiple seeds using:

```python
GREAT_RUN_SEEDS = [42, 43, 44]
```

Baseline models are reported from a single run by default, while the proposed GrEAt model is summarized across three runs.

## Notes on anonymization

This repository is prepared for anonymous review. It does not include author names, institutional identifiers, or non-anonymous acknowledgments. Replace anonymous placeholders only after the review period if required.

## Citation

If this code is used, please cite the accompanying anonymous manuscript once citation information becomes available.

```bibtex
@inproceedings{anonymous_great,
  title     = {GrEAt: Generalizable Energy-Based Anomaly Detection for Tabular Electronic Health Records},
  author    = {Anonymous},
  booktitle = {Anonymous submission},
  year      = {2026}
}
```

## License

This repository is released for research and reproducibility purposes. Add the final license file required by the target venue or host repository.
