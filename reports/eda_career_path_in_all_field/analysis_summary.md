# Exploratory data analysis summary

Source: `data/raw/career_path_in_all_field.csv`

## Dataset overview

- Rows: 9,000
- Columns: 17
- Exact duplicate rows beyond the first copy: 0
- Feature profiles associated with more than one target: 0
- Target: `Career`
- Target classes: 90

## Missing values

No missing values were found after trimming whitespace-only text.

## Strongest measured associations with the target

- `Coding_Skills`: 0.115 (correlation ratio eta)
- `Analytical_Skills`: 0.111 (correlation ratio eta)
- `Networking_Skills`: 0.108 (correlation ratio eta)
- `Communication_Skills`: 0.106 (correlation ratio eta)
- `Problem_Solving_Skills`: 0.106 (correlation ratio eta)
- `Teamwork_Skills`: 0.105 (correlation ratio eta)
- `Leadership_Positions`: 0.105 (correlation ratio eta)
- `Presentation_Skills`: 0.104 (correlation ratio eta)
- `Internships`: 0.102 (correlation ratio eta)
- `Extracurricular_Activities`: 0.101 (correlation ratio eta)

## Interpretation notes

- Pearson and Spearman correlations describe numeric-to-numeric relationships.
- The mixed matrix uses absolute Spearman correlation for two numeric fields, bias-corrected Cramér's V for two categorical fields, and correlation ratio eta for numeric-to-categorical relationships.
- Association values show relationship strength, not direction or causation.
- High-cardinality categorical features can appear stronger because they contain many distinct groups; interpret them with their unique-value counts.
- IQR outlier flags are screening indicators, not automatic data errors.
