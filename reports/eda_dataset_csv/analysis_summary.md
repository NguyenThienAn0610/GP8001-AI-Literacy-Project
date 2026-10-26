# Exploratory data analysis summary

Source: `data/raw/dataset.csv`

## Dataset overview

- Rows: 300,000
- Columns: 15
- Exact duplicate rows beyond the first copy: 3
- Feature profiles associated with more than one target: 0
- Target: `Job_Sector`
- Target classes: 8
- Rows with a usable target label: 156,644 (52.2%)

## Missing values

- `Job_Sector`: 143,356 rows (47.8%)

## Strongest measured associations with the target

- `Salary`: 0.009 (correlation ratio eta)
- `Years_Since_Graduation`: 0.008 (correlation ratio eta)
- `Age`: 0.008 (correlation ratio eta)
- `GPA`: 0.005 (correlation ratio eta)
- `Gender`: 0.004 (bias corrected cramers v)
- `Visa_Type`: 0.003 (bias corrected cramers v)
- `Field_of_Study`: 0.003 (bias corrected cramers v)
- `Education_Level`: 0.003 (bias corrected cramers v)
- `Country_of_Origin`: 0.001 (bias corrected cramers v)
- `Language_Proficiency`: 0.000 (bias corrected cramers v)

## Interpretation notes

- Pearson and Spearman correlations describe numeric-to-numeric relationships.
- The mixed matrix uses absolute Spearman correlation for two numeric fields, bias-corrected Cramér's V for two categorical fields, and correlation ratio eta for numeric-to-categorical relationships.
- Association values show relationship strength, not direction or causation.
- Target associations are calculated only on rows with a non-missing target.
- High-cardinality categorical features can appear stronger because they contain many distinct groups; interpret them with their unique-value counts.
- IQR outlier flags are screening indicators, not automatic data errors.
