# Exploratory data analysis summary

Source: `data/raw/Career_Path_Dataset_with_Career.csv`

## Dataset overview

- Rows: 15,000
- Columns: 19
- Exact duplicate rows beyond the first copy: 0
- Feature profiles associated with more than one target: 0
- Target: `Suggested_Career_Path`
- Target classes: 22

## Missing values

- `Certifications`: 5,099 rows (34.0%)
- `Past_Jobs_Internships`: 2,538 rows (16.9%)
- `Achievements`: 2,532 rows (16.9%)

## Strongest measured associations with the target

- `Fields_of_Interest`: 0.999 (bias corrected cramers v)
- `Grade_CGPA_Percentage`: 0.047 (correlation ratio eta)
- `Age`: 0.044 (correlation ratio eta)
- `Willing_to_Relocate`: 0.026 (bias corrected cramers v)
- `Technical_Skills`: 0.025 (bias corrected cramers v)
- `Achievements`: 0.018 (bias corrected cramers v)
- `Gender`: 0.015 (bias corrected cramers v)
- `Stream`: 0.010 (bias corrected cramers v)
- `Current_Academic_Level`: 0.008 (bias corrected cramers v)
- `Location`: 0.008 (bias corrected cramers v)

## Interpretation notes

- Pearson and Spearman correlations describe numeric-to-numeric relationships.
- The mixed matrix uses absolute Spearman correlation for two numeric fields, bias-corrected Cramér's V for two categorical fields, and correlation ratio eta for numeric-to-categorical relationships.
- Association values show relationship strength, not direction or causation.
- High-cardinality categorical features can appear stronger because they contain many distinct groups; interpret them with their unique-value counts.
- IQR outlier flags are screening indicators, not automatic data errors.
- The full mixed matrix omits high-cardinality columns to keep it interpretable: `Technical_Skills`, `Soft_Skills`. Their target associations are still reported.
