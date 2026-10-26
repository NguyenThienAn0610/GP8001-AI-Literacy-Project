# `dataset.csv` evaluation summary

## Decision

This dataset is **not better for the career-recommendation task**. It is larger
and its eight `Job_Sector` classes are balanced, but the available input
features contain essentially no reproducible signal about the target.

## Evaluation design

- Raw rows: 300,000
- Rows with a known `Job_Sector`: 156,644 (52.2%)
- Target classes: 8
- Validation: stratified five-fold out-of-fold predictions
- Features used: 12 pre-outcome demographic, education, and experience fields
- Deliberately excluded: `Employment_Status` and `Salary` (post-outcome fields)
- Uniform-random baselines: top-1 12.5%, top-2 25.0%, top-3 37.5%

## Results

| Model | Top-1 | Macro F1 | Top-2 | Top-3 | Top-1 / chance |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 12.441% | 12.299% | 25.110% | 37.646% | 0.995x |
| Extra Trees | 12.528% | 12.512% | 24.929% | 37.428% | 1.002x |
| Uniform random | 12.500% | — | 25.000% | 37.500% | 1.000x |

The strongest single measured feature association with `Job_Sector` among
labeled rows is only 0.009. `Field_of_Study` has a bias-corrected Cramér's V of
0.003, so it does not meaningfully determine the sector either.

## Important data-quality findings

- `Job_Sector` is missing for 143,356 rows (47.8%) and is present only when
  `Employment_Status` is `Employed`. Models therefore learn only from people
  already employed and cannot be assumed to generalize to unemployed students.
- The target is extremely uniform and almost independent of every feature,
  which is consistent with synthetic random assignment.
- There are only three exact duplicate rows, and no other columns are missing.
- The label describes a broad sector, not a specific career or role, so it is
  also less aligned with the original application's output.

## Comparison with the earlier datasets

- Original data with `Fields_of_Interest`: 32.82% accuracy over 22 classes, but
  that feature dominates the prediction and likely encodes the label-generation
  rule.
- Original data without `Fields_of_Interest`: 4.65% versus 4.55% random chance.
- `career_path_in_all_field.csv`: approximately 1.19% versus 1.11% chance over
  90 classes.
- `dataset.csv`: 12.53% versus 12.50% chance over only eight classes.

Raw accuracy is not comparable across different class counts. Relative to each
dataset's chance baseline, all three alternatives without the dominant field
feature are effectively non-predictive.
