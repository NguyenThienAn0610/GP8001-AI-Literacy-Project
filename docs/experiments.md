# Experiments and result interpretation

[Back to the project README](../README.md).

## Why Top-1 is near 33% while Top-3 is near 96.5%

Top-1 asks whether the first suggestion matches the recorded career. Top-3 asks
only whether that career appears anywhere in a three-item shortlist.

The dataset has seven interest groups. Six contain exactly three careers, and
`Software Development` contains four. For example, the 2,140 Data Science
profiles have almost equally frequent labels: Data Analyst 719 (33.6%), Data
Scientist 716 (33.5%), and Business Analyst 705 (32.9%). Recognising this group
provides a strong shortlist, while picking one label requires useful evidence
that distinguishes careers within the group.

A lookup based only on training-fold interest frequencies reaches **32.99%
Top-1 and 96.61% Top-3** on the same five folds. Its macro-F1 is only **0.1578**,
because its first choices cover only seven of the 22 careers. This shows how
much of the high shortlist coverage follows from the dataset's structure.

Software Development accounts for 14.67% of rows. If careers were equally likely
within each group and the group were known, the illustrative Top-3 calculation
would be `85.33% * 100% + 14.67% * 75% = 96.33%`. This is an explanation of the
pattern, not a proven accuracy ceiling. Balanced labels alone do not establish
that other features contain no signal. The tested pipelines show limited
success at distinguishing careers within interest groups.

These scores measure agreement with dataset labels, not real-world career
suitability. More informative data and external validation are needed before
treating the output as reliable personal advice. Reproduce the diagnostic with:

```powershell
.venv\Scripts\python.exe src/analyze_recommendation_gap.py
```

The script writes `reports/recommendation_gap.json` and CSVs with group counts,
fold metrics, and held-out predictions.

## Advanced model experiments

Two additional nonlinear pipelines were evaluated with the same stratified
five-fold split and seed 42. Logistic Regression was rerun and exactly
reproduced its saved metrics.

| Pipeline | Top-1 | Macro-F1 | Top-3 |
|---|---:|---:|---:|
| Logistic Regression | 32.82% | 0.3248 | 96.49% |
| Histogram Gradient Boosting | 31.52% | 0.3105 | 96.46% |
| MLP neural network | 31.95% | 0.2766 | 96.55% |

Gradient boosting adds trees sequentially to reduce prediction errors. The
tested configuration uses up to 150 rounds, 15 leaves per tree, and binary
unigram profile features. The MLP uses two hidden layers with 64 and 32 ReLU
units, Adam optimisation, and TF-IDF unigram features. Both use early stopping
with an internal 10% validation split and finished below their iteration limits.

These are fixed exploratory pipelines, not an exhaustive tuning exercise or a
statistical superiority test. Preprocessing, class weighting, and effective
training sizes differ between pipelines. Preprocessing is fitted on the outer
training fold before the estimator's internal validation split; the outer test
fold remains isolated throughout. Neither advanced configuration improved
Top-1 or macro-F1 over Logistic Regression. The neural network's small Top-3
increase does not establish a meaningful advantage.

```powershell
.venv\Scripts\python.exe src/benchmark_advanced_models.py
```

The benchmark writes `reports/advanced_model_comparison.csv`, per-fold metrics,
out-of-fold predictions, and `reports/advanced_benchmark.json` with exact
settings, dataset hash, package versions, timings, and warnings. It does not
replace the saved models used by `predict.cmd`.

## Deep learning architecture and learning-rate search

A separate GPU experiment tested **18 configurations across five outer folds**
(90 tuning trials), followed by refits of each family's selected configuration.
It used PyTorch 2.13.0 with CUDA 13.0 on an NVIDIA RTX 2000 Ada Generation GPU.
The original Logistic Regression and MLP were rerun on the same folds and
exactly reproduced their previously reported accuracy, macro-F1, and Top-3.

| Approach | Top-1 | Macro-F1 | Top-3 |
|---|---:|---:|---:|
| Existing Logistic Regression | **32.82%** | **0.3248** | 96.49% |
| Original MLP, 64/32 hidden units | 31.95% | 0.2766 | **96.55%** |
| Tuned deep MLP | 32.32% | 0.2439 | 95.48% |
| Tuned residual network | 31.61% | 0.2902 | 96.31% |
| Tuned field Transformer | 32.23% | 0.2353 | 96.47% |
| Network family selected by inner validation | 32.07% | 0.2547 | 96.45% |
| Equal average of the three tuned families | 32.05% | 0.2952 | 96.33% |

The deep MLP improved Top-1 over the original MLP by **0.37 percentage points**,
but lost macro-F1 and Top-3 coverage. It remained **0.50 points below Logistic
Regression**, with lower Top-1 on all five test folds. The new neural approaches
did not improve the best existing first-choice accuracy or macro-F1. These are
descriptive results, not a statistical superiority test or an accuracy ceiling.
The neural models are evaluation experiments; `predict.cmd` uses the saved scikit-learn pipelines.

The search covered:

- **Deep MLP:** hidden widths 256/128/64, 512/256/128, or
  512/256/128/64/32, using LayerNorm, GELU and dropout. Parameter counts range
  from 68,310 to 226,934.
- **Residual network:** 128 or 256 hidden units, with two, three, or four
  residual blocks. Each block has two linear layers and a skip connection.
  Parameter counts range from 147,862 to 821,270.
- **Field Transformer:** one learned token for each of the 18 original fields,
  plus a classification token. Four attention heads, token widths 32 or 64,
  and two or four Transformer blocks. Parameter counts range from 29,878 to
  108,886. This is a custom tabular architecture, not a pretrained model.
- **Training settings:** AdamW with initial learning rates 0.0003, 0.001 or
  0.003; dropout 10–30%; weight decay 0.001, 0.01 or 0.1; batches of 256 or
  512; constant rates, reduction on a validation-loss plateau, or cosine decay.
  The search also compared binary and TF-IDF unigram inputs, and included
  a variant with label smoothing of 0.05. This is a structured selection of
  configurations, not every possible combination.

Each outer fold contains 12,000 training profiles and 3,000 test profiles.
An inner stratified split uses 9,600 profiles for training and 2,400 for
validation. Preprocessing fits only those 9,600 profiles during the search.
Validation Top-1 selects the configuration and training epoch; macro-F1 and
log loss break ties. Early stopping has a patience of 12 epochs and a maximum
of 60 epochs. No trial reached the maximum.

For the final fold evaluation, preprocessing refits on all 12,000 outer
training profiles. The selected network trains for the inner-selected epoch
count, replaying the learning rates chosen inside the training fold. The
3,000 outer test profiles do not influence configuration, epoch, or learning
rate selection. Each tuned family can select different settings in different
folds, so its result evaluates that selection procedure. The ensemble averages
the three family predictions with equal weights fixed before evaluation.
The dataset was already explored in previous experiments; this is not new
external validation.

Install the optional GPU dependency and reproduce the experiment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-deep-learning.txt --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe src/benchmark_deep_learning.py --smoke-test
.venv\Scripts\python.exe src/benchmark_deep_learning.py --report-dir reports/deep_learning_search_rerun
.venv\Scripts\python.exe src/summarize_deep_learning.py --report-dir reports/deep_learning_search_rerun
```

The CUDA wheel requires a compatible NVIDIA driver. Use the appropriate
official PyTorch wheel for other hardware; the script also accepts `--device cpu`.
Existing experiment directories are protected against accidental replacement.

The completed run is in `reports/deep_learning_search/`. `model_comparison.csv`
contains the final scores, `selected_configurations.csv` identifies each fold's
settings, and `hyperparameter_summary.csv` reports **inner validation** scores
for all 18 configurations. `trials.json` contains every epoch's learning rate,
training loss and validation scores. The directory also includes split
assignments, out-of-fold predictions, paired accuracy differences and
`validation.json`. The audit recomputes scores from all 15,000 predictions per
approach and checks that selection uses only the inner validation results.

## Feature relevance experiments

Run these from the project root after installing the base requirements:

```powershell
.venv\Scripts\python.exe src/analyze_feature_ablation.py --report-dir reports/ablation_rerun
.venv\Scripts\python.exe src/analyze_feature_permutation.py --report-dir reports/permutation_rerun
```

Ablation fits four separate pipelines: Logistic Regression with all features,
interest only, and without interest; and Histogram Gradient Boosting without
interest. Excluded fields are removed before tokenization or vocabulary fitting.
Removing interest reduced Top-1 to about 4.65% for Logistic Regression and 4.68%
for boosting, close to the uniform 22-class reference of 4.55%.

Permutation measures how much the fitted Logistic Regression relies on each
field. Non-interest fields are shuffled within interest groups; the joint
shuffle preserves relationships among those fields. Interest is shuffled
globally. Jointly shuffling the other 17 fields reduced Top-1 by about 0.93
percentage points. These results suggest limited additional predictive signal
in this dataset, not that education or skills are unimportant to real careers.

Saved outputs are `reports/feature_ablation*` and `reports/feature_permutation*`.
Use a separate report directory to retain previous runs.

## Audit saved deep learning results

```powershell
.venv\Scripts\python.exe src/summarize_deep_learning.py --output-dir reports/deep_learning_audit
```

The audit checks the dataset hash, disjoint split assignments, configuration
selection, and metrics recomputed from saved probabilities. It writes derived
summary files only after these checks pass. `--output-dir` keeps the original
experiment files intact; without it, derived summaries are written in the
experiment directory.

Source provenance is reported separately as `source_matches_recorded`. The
historical run predates the code refactor, so its original script hash will not
match the current code. That does not prevent auditing the saved results. Use
`--strict-source` when an exact source match is required. New runs record hashes
for the benchmark and its shared model, preprocessing, and evaluation modules.

## Code organization

- `career_ml.py`: input schema, preprocessing, production model factories, and
  labelled-data validation. The transformer stays here so existing saved models
  remain loadable.
- `evaluation.py`: classification scores, validation ranking, and fold summaries.
- `experiment_models.py`: fixed boosting and original MLP configurations.
- `neural_models.py`: PyTorch architectures, search space, training, and refitting.
- `benchmark_*.py` and `analyze_*.py`: command-line experiment orchestration.

All paths in commands are relative to the project root. The benchmark scripts
write evaluation results; only `train_models.py` saves deployment pipelines.
