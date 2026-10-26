"""Generate a reproducible exploratory analysis of the raw career dataset.

The script writes machine-readable summary tables and publication-ready PNG
figures without changing the source CSV.  It supports numeric, categorical,
and comma-separated multi-value columns.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


DEFAULT_DATA_PATH = Path("data/raw/Career_Path_Dataset_with_Career.csv")
DEFAULT_OUTPUT_DIR = Path("reports/eda")
DEFAULT_TARGET = "Suggested_Career_Path"
DEFAULT_MULTI_VALUE_COLUMNS = [
    "Technical_Skills",
    "Soft_Skills",
    "Languages_Known",
    "Certifications",
]
MISSING_LABEL = "<Missing>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--multi-value-columns",
        nargs="*",
        default=DEFAULT_MULTI_VALUE_COLUMNS,
        help="Columns containing comma-separated values.",
    )
    parser.add_argument(
        "--top-categories",
        type=int,
        default=15,
        help="Maximum categories or tokens shown in each distribution plot.",
    )
    parser.add_argument(
        "--association-max-categories",
        type=int,
        default=100,
        help=(
            "Maximum distinct levels for a categorical column in the full "
            "mixed-association matrix. Target associations are calculated for "
            "all columns regardless of this limit. Use 0 for no limit."
        ),
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_categories < 1:
        raise ValueError("--top-categories must be at least 1.")
    if args.association_max_categories < 0:
        raise ValueError("--association-max-categories cannot be negative.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError(
            "Duplicate columns remain after trimming header whitespace: "
            + ", ".join(duplicates)
        )
    return frame


def normalize_text_for_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim text and treat empty strings as missing in an analysis-only copy."""
    cleaned = frame.copy()
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)
    return cleaned


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()


def column_profile(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    target: str,
    multi_value_columns: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in cleaned.columns:
        series = cleaned[column]
        raw_series = raw[column]
        if column == target:
            role = "target"
        elif column in multi_value_columns:
            role = "multi-value categorical"
        elif pd.api.types.is_numeric_dtype(series):
            role = "numeric"
        else:
            role = "categorical"

        whitespace_only = 0
        leading_or_trailing_whitespace = 0
        if pd.api.types.is_object_dtype(raw_series) or isinstance(
            raw_series.dtype, pd.StringDtype
        ):
            raw_text = raw_series.dropna().astype(str)
            whitespace_only = int(raw_text.str.fullmatch(r"\s*").sum())
            leading_or_trailing_whitespace = int(
                raw_text.ne(raw_text.str.strip()).sum()
            )

        rows.append(
            {
                "column": column,
                "pandas_dtype": str(series.dtype),
                "role": role,
                "rows": len(series),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "unique_non_missing": int(series.nunique(dropna=True)),
                "constant": bool(series.nunique(dropna=True) <= 1),
                "whitespace_only_values": whitespace_only,
                "values_with_edge_whitespace": leading_or_trailing_whitespace,
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(frame: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    if not numeric:
        return pd.DataFrame()
    summary = frame[numeric].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    summary["missing"] = frame[numeric].isna().sum()
    summary["missing_pct"] = frame[numeric].isna().mean()
    summary["skew"] = frame[numeric].skew(numeric_only=True)
    summary["kurtosis"] = frame[numeric].kurtosis(numeric_only=True)
    return summary.reset_index(names="column")


def iqr_outlier_summary(frame: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric:
        values = frame[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": outlier_count,
                "outlier_pct_of_non_missing": (
                    outlier_count / len(values) if len(values) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def categorical_value_counts(
    frame: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        counts = frame[column].fillna(MISSING_LABEL).astype(str).value_counts(dropna=False)
        for rank, (value, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "column": column,
                    "rank": rank,
                    "value": value,
                    "count": int(count),
                    "pct": float(count / len(frame)) if len(frame) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def split_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return [MISSING_LABEL]
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    return tokens or [MISSING_LABEL]


def token_counts(
    frame: pd.DataFrame, multi_value_columns: Iterable[str], target: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for column in multi_value_columns:
        if column not in frame.columns:
            continue
        exploded = frame[[column] + ([target] if target and target in frame else [])].copy()
        exploded["token"] = exploded[column].map(split_tokens)
        exploded = exploded.explode("token", ignore_index=True)

        counts = exploded["token"].value_counts()
        for rank, (token, count) in enumerate(counts.items(), start=1):
            overall_rows.append(
                {
                    "column": column,
                    "rank": rank,
                    "token": token,
                    "count": int(count),
                    "pct_of_rows": float(count / len(frame)) if len(frame) else np.nan,
                }
            )

        if target and target in frame:
            exploded[target] = exploded[target].fillna(MISSING_LABEL)
            grouped = (
                exploded.groupby([target, "token"], dropna=False)
                .size()
                .rename("count")
                .reset_index()
            )
            target_sizes = frame[target].fillna(MISSING_LABEL).value_counts()
            for target_value, token, count in grouped.itertuples(index=False, name=None):
                denominator = target_sizes.get(target_value, 0)
                target_rows.append(
                    {
                        "column": column,
                        "target_value": target_value,
                        "token": token,
                        "count": int(count),
                        "pct_within_target": (
                            float(count / denominator) if denominator else np.nan
                        ),
                    }
                )
    return pd.DataFrame(overall_rows), pd.DataFrame(target_rows)


def bias_corrected_cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Return bias-corrected Cramer's V without requiring SciPy."""
    valid = left.notna() | right.notna()
    x = left.loc[valid].fillna(MISSING_LABEL).astype(str)
    y = right.loc[valid].fillna(MISSING_LABEL).astype(str)
    observed = pd.crosstab(x, y).to_numpy(dtype=float)
    n = observed.sum()
    if n <= 1 or min(observed.shape) <= 1:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    nonzero = expected > 0
    chi2 = np.sum(((observed - expected) ** 2)[nonzero] / expected[nonzero])
    phi2 = chi2 / n
    rows, columns = observed.shape
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    return float(math.sqrt(phi2_corrected / denominator)) if denominator > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Return eta, the strength of association between categories and numbers."""
    valid = values.notna()
    if not valid.any():
        return np.nan
    y = values.loc[valid].astype(float)
    groups = categories.loc[valid].fillna(MISSING_LABEL).astype(str)
    grand_mean = y.mean()
    total = float(((y - grand_mean) ** 2).sum())
    if total == 0:
        return 0.0
    grouped = pd.DataFrame({"group": groups, "value": y}).groupby("group")["value"]
    between = float(
        sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in grouped)
    )
    return float(math.sqrt(max(0.0, between / total)))


def mixed_association_matrix(
    frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    numeric = set(frame[columns].select_dtypes(include=np.number).columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            if left in numeric and right in numeric:
                value = frame[left].rank().corr(
                    frame[right].rank(), method="pearson"
                )
                value = abs(float(value)) if pd.notna(value) else np.nan
            elif left not in numeric and right not in numeric:
                value = bias_corrected_cramers_v(frame[left], frame[right])
            elif left in numeric:
                value = correlation_ratio(frame[right], frame[left])
            else:
                value = correlation_ratio(frame[left], frame[right])
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def target_associations(
    frame: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    if target not in frame.columns:
        return pd.DataFrame()
    # Missing labels are not a target class. Including them can create a large,
    # spurious association whenever label availability depends on another field
    # (for example, Job_Sector is recorded only for employed respondents).
    labeled = frame.loc[frame[target].notna()]
    rows: list[dict[str, object]] = []
    for column in numeric:
        if column == target:
            continue
        rows.append(
            {
                "feature": column,
                "feature_type": "numeric",
                "measure": "correlation_ratio_eta",
                "association": correlation_ratio(labeled[target], labeled[column]),
                "unique_values": int(frame[column].nunique(dropna=True)),
            }
        )
    for column in categorical:
        if column == target:
            continue
        rows.append(
            {
                "feature": column,
                "feature_type": "categorical",
                "measure": "bias_corrected_cramers_v",
                "association": bias_corrected_cramers_v(labeled[column], labeled[target]),
                "unique_values": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("association", ascending=False)


def missingness_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame.isna().apply(
        lambda row: ", ".join(frame.columns[row].tolist()) or "No missing values",
        axis=1,
    )
    counts = labels.value_counts().rename_axis("missing_columns").reset_index(name="count")
    counts["pct"] = counts["count"] / len(frame) if len(frame) else np.nan
    return counts


def duplicate_diagnostics(frame: pd.DataFrame, target: str) -> dict[str, int]:
    duplicate_rows = int(frame.duplicated(keep=False).sum())
    duplicate_rows_beyond_first = int(frame.duplicated().sum())
    result = {
        "rows_in_duplicate_groups": duplicate_rows,
        "duplicate_rows_beyond_first": duplicate_rows_beyond_first,
        "feature_rows_in_duplicate_groups": 0,
        "feature_profiles_with_conflicting_targets": 0,
    }
    if target in frame.columns:
        features = [column for column in frame.columns if column != target]
        result["feature_rows_in_duplicate_groups"] = int(
            frame.duplicated(subset=features, keep=False).sum()
        )
        target_counts = frame.groupby(features, dropna=False)[target].nunique(dropna=False)
        result["feature_profiles_with_conflicting_targets"] = int((target_counts > 1).sum())
    return result


def save_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, encoding="utf-8-sig")


def finish_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_missingness(profile: pd.DataFrame, path: Path, dpi: int) -> None:
    data = profile.loc[profile["missing"] > 0].sort_values("missing_pct")
    fig_height = max(3.2, 0.45 * max(len(data), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    if data.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=13)
        ax.set_axis_off()
    else:
        sns.barplot(data=data, x="missing_pct", y="column", color="#D97706", ax=ax)
        ax.set_title("Missing values by column")
        ax.set_xlabel("Share of rows")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", labels=[f"{v:.1%}" for v in data["missing_pct"]])
    finish_figure(fig, path, dpi)


def plot_numeric_distributions(
    frame: pd.DataFrame, numeric: list[str], path: Path, dpi: int
) -> None:
    if not numeric:
        return
    columns = min(2, len(numeric))
    rows = math.ceil(len(numeric) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 4.2 * rows), squeeze=False)
    for ax, column in zip(axes.flat, numeric):
        sns.histplot(frame[column].dropna(), bins="auto", kde=True, color="#2563EB", ax=ax)
        ax.axvline(frame[column].median(), color="#B91C1C", linestyle="--", label="Median")
        ax.set_title(column.replace("_", " "))
        ax.set_ylabel("Rows")
        ax.legend(frameon=False)
    for ax in axes.flat[len(numeric) :]:
        ax.set_visible(False)
    fig.suptitle("Numeric distributions", fontsize=15, y=1.01)
    fig.tight_layout()
    finish_figure(fig, path, dpi)


def plot_target_distribution(
    frame: pd.DataFrame, target: str, path: Path, dpi: int
) -> None:
    if target not in frame.columns:
        return
    counts = frame[target].fillna(MISSING_LABEL).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(counts) + 1.8)))
    ax.barh(counts.index.astype(str), counts.values, color="#0F766E")
    ax.set_title(f"{target.replace('_', ' ')} distribution")
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    ax.bar_label(ax.containers[0], padding=3, fontsize=8)
    finish_figure(fig, path, dpi)


def plot_categorical_pages(
    frame: pd.DataFrame,
    columns: list[str],
    output_dir: Path,
    top_n: int,
    dpi: int,
) -> None:
    per_page = 6
    for page_start in range(0, len(columns), per_page):
        page_columns = columns[page_start : page_start + per_page]
        fig, axes = plt.subplots(3, 2, figsize=(16, 14), squeeze=False)
        for ax, column in zip(axes.flat, page_columns):
            counts = frame[column].fillna(MISSING_LABEL).astype(str).value_counts()
            shown = counts.head(top_n).sort_values()
            omitted = counts.iloc[top_n:].sum()
            if omitted:
                shown.loc[f"Other ({len(counts) - top_n} levels)"] = omitted
                shown = shown.sort_values()
            ax.barh(shown.index, shown.values, color="#4F46E5")
            ax.set_title(column.replace("_", " "))
            ax.set_xlabel("Rows")
            ax.tick_params(axis="y", labelsize=8)
        for ax in axes.flat[len(page_columns) :]:
            ax.set_visible(False)
        fig.suptitle("Categorical distributions", fontsize=16, y=1.005)
        fig.tight_layout()
        page = page_start // per_page + 1
        finish_figure(fig, output_dir / f"categorical_distributions_{page:02d}.png", dpi)


def plot_token_frequencies(
    token_table: pd.DataFrame, path: Path, top_n: int, dpi: int
) -> None:
    if token_table.empty:
        return
    columns = token_table["column"].drop_duplicates().tolist()
    grid_columns = min(2, len(columns))
    grid_rows = math.ceil(len(columns) / grid_columns)
    fig, axes = plt.subplots(
        grid_rows,
        grid_columns,
        figsize=(8 * grid_columns, 4.8 * grid_rows),
        squeeze=False,
    )
    for ax, column in zip(axes.flat, columns):
        data = token_table.loc[token_table["column"] == column].head(top_n).sort_values("count")
        ax.barh(data["token"], data["count"], color="#0891B2")
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("Rows containing token")
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes.flat[len(columns) :]:
        ax.set_visible(False)
    fig.suptitle("Multi-value field token frequencies", fontsize=16, y=1.005)
    fig.tight_layout()
    finish_figure(fig, path, dpi)


def plot_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    dpi: int,
    *,
    center: float | None = None,
    value_range: tuple[float, float] | None = None,
) -> None:
    if matrix.empty:
        return
    display_matrix = matrix.copy()
    display_matrix.index = [label.replace("_", " ") for label in matrix.index]
    display_matrix.columns = [label.replace("_", " ") for label in matrix.columns]
    size = max(6, min(18, 0.62 * len(matrix.columns) + 3))
    fig, ax = plt.subplots(figsize=(size, size))
    annotate = len(matrix.columns) <= 12
    sns.heatmap(
        display_matrix,
        cmap="vlag" if center is not None else "YlGnBu",
        center=center,
        vmin=value_range[0] if value_range else None,
        vmax=value_range[1] if value_range else None,
        annot=annotate,
        fmt=".2f",
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, pad=14)
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    finish_figure(fig, path, dpi)


def plot_target_associations(table: pd.DataFrame, path: Path, dpi: int) -> None:
    if table.empty:
        return
    data = table.sort_values("association").copy()
    data["display_feature"] = data["feature"].str.replace("_", " ")
    colors = ["#2563EB" if kind == "numeric" else "#7C3AED" for kind in data["feature_type"]]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(data) + 1.8)))
    ax.barh(data["display_feature"], data["association"], color=colors)
    ax.set_xlim(0, max(1.0, float(data["association"].max()) * 1.08))
    ax.set_title("Feature association with target")
    ax.set_xlabel("Association strength (0 to 1)")
    ax.set_ylabel("")
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=8)
    ax.legend(
        handles=[
            Patch(color="#2563EB", label="Numeric"),
            Patch(color="#7C3AED", label="Categorical"),
        ],
        frameon=False,
        loc="lower right",
    )
    finish_figure(fig, path, dpi)


def plot_numeric_by_target(
    frame: pd.DataFrame,
    numeric: list[str],
    target: str,
    path: Path,
    dpi: int,
) -> None:
    if target not in frame.columns or not numeric:
        return
    fig, axes = plt.subplots(
        len(numeric),
        1,
        figsize=(13, max(6, 0.42 * frame[target].nunique() * len(numeric))),
        squeeze=False,
    )
    order = frame[target].fillna(MISSING_LABEL).value_counts().index
    plot_frame = frame.copy()
    plot_frame[target] = plot_frame[target].fillna(MISSING_LABEL)
    for ax, column in zip(axes.flat, numeric):
        sns.boxplot(
            data=plot_frame,
            x=column,
            y=target,
            order=order,
            color="#93C5FD",
            showfliers=False,
            ax=ax,
        )
        ax.set_title(f"{column.replace('_', ' ')} by target")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    finish_figure(fig, path, dpi)


def plot_target_crosstab(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    path: Path,
    dpi: int,
) -> None:
    if feature not in frame.columns or target not in frame.columns:
        return
    table = pd.crosstab(
        frame[feature].fillna(MISSING_LABEL),
        frame[target].fillna(MISSING_LABEL),
        normalize="index",
    )
    fig_width = max(12, 0.58 * len(table.columns) + 4)
    fig_height = max(4.5, 0.52 * len(table.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(table, cmap="Blues", annot=False, vmin=0, vmax=table.to_numpy().max(), ax=ax)
    ax.set_title(f"Target mix within {feature.replace('_', ' ')}")
    ax.set_xlabel(target.replace("_", " "))
    ax.set_ylabel(feature.replace("_", " "))
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    finish_figure(fig, path, dpi)


def write_markdown_summary(
    path: Path,
    data_path: Path,
    frame: pd.DataFrame,
    profile: pd.DataFrame,
    duplicates: dict[str, int],
    associations: pd.DataFrame,
    target: str,
    excluded_association_columns: list[str],
) -> None:
    missing = profile.loc[profile["missing"] > 0].sort_values("missing_pct", ascending=False)
    lines = [
        "# Exploratory data analysis summary",
        "",
        f"Source: `{data_path.as_posix()}`",
        "",
        "## Dataset overview",
        "",
        f"- Rows: {len(frame):,}",
        f"- Columns: {len(frame.columns):,}",
        f"- Exact duplicate rows beyond the first copy: {duplicates['duplicate_rows_beyond_first']:,}",
        (
            "- Feature profiles associated with more than one target: "
            f"{duplicates['feature_profiles_with_conflicting_targets']:,}"
        ),
    ]
    if target in frame.columns:
        labeled_rows = int(frame[target].notna().sum())
        lines.extend(
            [
                f"- Target: `{target}`",
                f"- Target classes: {frame[target].nunique(dropna=True):,}",
                f"- Rows with a usable target label: {labeled_rows:,} ({labeled_rows / len(frame):.1%})",
            ]
        )

    lines.extend(["", "## Missing values", ""])
    if missing.empty:
        lines.append("No missing values were found after trimming whitespace-only text.")
    else:
        for row in missing.itertuples(index=False):
            lines.append(f"- `{row.column}`: {row.missing:,} rows ({row.missing_pct:.1%})")

    if not associations.empty:
        lines.extend(["", "## Strongest measured associations with the target", ""])
        for row in associations.head(10).itertuples(index=False):
            lines.append(
                f"- `{row.feature}`: {row.association:.3f} "
                f"({row.measure.replace('_', ' ')})"
            )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Pearson and Spearman correlations describe numeric-to-numeric relationships.",
            "- The mixed matrix uses absolute Spearman correlation for two numeric fields, "
            "bias-corrected Cramér's V for two categorical fields, and correlation ratio "
            "eta for numeric-to-categorical relationships.",
            "- Association values show relationship strength, not direction or causation.",
            "- Target associations are calculated only on rows with a non-missing target.",
            "- High-cardinality categorical features can appear stronger because they contain "
            "many distinct groups; interpret them with their unique-value counts.",
            "- IQR outlier flags are screening indicators, not automatic data errors.",
        ]
    )
    if excluded_association_columns:
        joined = ", ".join(f"`{column}`" for column in excluded_association_columns)
        lines.append(
            "- The full mixed matrix omits high-cardinality columns to keep it interpretable: "
            f"{joined}. Their target associations are still reported."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    raw = load_data(args.data)
    frame = normalize_text_for_analysis(raw)
    if args.target and args.target not in frame.columns:
        raise ValueError(f"Target column not found: {args.target}")

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    multi_value = {
        column for column in args.multi_value_columns if column in frame.columns
    }
    missing_multi_value = sorted(set(args.multi_value_columns) - set(frame.columns))
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = categorical_columns(frame)

    profile = column_profile(raw, frame, args.target, multi_value)
    numeric_stats = numeric_summary(frame, numeric)
    outliers = iqr_outlier_summary(frame, numeric)
    category_counts = categorical_value_counts(frame, categorical)
    overall_tokens, target_tokens = token_counts(
        frame, sorted(multi_value), args.target
    )
    pearson = frame[numeric].corr(method="pearson") if numeric else pd.DataFrame()
    # Spearman correlation is Pearson correlation applied to ranks. Expressing
    # it this way avoids pandas' optional runtime dependency on SciPy.
    spearman = (
        frame[numeric].rank().corr(method="pearson") if numeric else pd.DataFrame()
    )

    association_columns: list[str] = []
    excluded_association_columns: list[str] = []
    for column in frame.columns:
        unique = frame[column].nunique(dropna=True)
        allowed = (
            pd.api.types.is_numeric_dtype(frame[column])
            or args.association_max_categories == 0
            or unique <= args.association_max_categories
        )
        if allowed:
            association_columns.append(column)
        else:
            excluded_association_columns.append(column)
    mixed = mixed_association_matrix(frame, association_columns)
    associations = target_associations(
        frame, args.target, numeric, categorical
    )
    missing_patterns = missingness_patterns(frame)
    duplicates = duplicate_diagnostics(frame, args.target)

    target_counts: dict[str, int] = {}
    if args.target in frame.columns:
        target_counts = {
            str(key): int(value)
            for key, value in frame[args.target]
            .fillna(MISSING_LABEL)
            .value_counts()
            .items()
        }
    overview = {
        "source": str(args.data),
        "rows": len(frame),
        "columns": len(frame.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "multi_value_columns": sorted(multi_value),
        "configured_multi_value_columns_not_found": missing_multi_value,
        "target": args.target,
        "target_class_counts": target_counts,
        "duplicates": duplicates,
        "association_matrix_columns": association_columns,
        "association_matrix_excluded_high_cardinality_columns": excluded_association_columns,
        "notes": {
            "source_unchanged": True,
            "text_analysis": "Leading/trailing whitespace trimmed; empty text treated as missing.",
            "mixed_association_matrix": (
                "Absolute Spearman for numeric-numeric, bias-corrected Cramer's V "
                "for categorical-categorical, and correlation ratio eta for mixed pairs."
            ),
        },
    }

    (output_dir / "dataset_overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_csv(profile, output_dir / "column_profile.csv")
    save_csv(numeric_stats, output_dir / "numeric_summary.csv")
    save_csv(outliers, output_dir / "numeric_iqr_outliers.csv")
    save_csv(category_counts, output_dir / "categorical_value_counts.csv")
    save_csv(overall_tokens, output_dir / "multi_value_token_counts.csv")
    save_csv(target_tokens, output_dir / "multi_value_tokens_by_target.csv")
    save_csv(missing_patterns, output_dir / "missingness_patterns.csv")
    save_csv(pearson, output_dir / "numeric_pearson_correlation.csv", index=True)
    save_csv(spearman, output_dir / "numeric_spearman_correlation.csv", index=True)
    save_csv(mixed, output_dir / "mixed_association_matrix.csv", index=True)
    save_csv(associations, output_dir / "target_associations.csv")

    plot_missingness(profile, plots_dir / "missing_values.png", args.dpi)
    plot_numeric_distributions(
        frame, numeric, plots_dir / "numeric_distributions.png", args.dpi
    )
    plot_target_distribution(
        frame, args.target, plots_dir / "target_distribution.png", args.dpi
    )
    ordinary_categories = [
        column
        for column in categorical
        if column not in multi_value and column != args.target
    ]
    plot_categorical_pages(
        frame,
        ordinary_categories,
        plots_dir,
        args.top_categories,
        args.dpi,
    )
    plot_token_frequencies(
        overall_tokens,
        plots_dir / "multi_value_token_frequencies.png",
        args.top_categories,
        args.dpi,
    )
    plot_heatmap(
        pearson,
        plots_dir / "numeric_pearson_correlation.png",
        "Numeric Pearson correlation",
        args.dpi,
        center=0,
        value_range=(-1, 1),
    )
    plot_heatmap(
        spearman,
        plots_dir / "numeric_spearman_correlation.png",
        "Numeric Spearman correlation",
        args.dpi,
        center=0,
        value_range=(-1, 1),
    )
    plot_heatmap(
        mixed,
        plots_dir / "mixed_association_heatmap.png",
        "Mixed-type association strength",
        args.dpi,
        value_range=(0, 1),
    )
    plot_target_associations(
        associations, plots_dir / "target_associations.png", args.dpi
    )
    plot_numeric_by_target(
        frame,
        numeric,
        args.target,
        plots_dir / "numeric_by_target.png",
        args.dpi,
    )
    if "Fields_of_Interest" in frame.columns:
        plot_target_crosstab(
            frame,
            "Fields_of_Interest",
            args.target,
            plots_dir / "target_mix_by_fields_of_interest.png",
            args.dpi,
        )

    write_markdown_summary(
        output_dir / "analysis_summary.md",
        args.data,
        frame,
        profile,
        duplicates,
        associations,
        args.target,
        excluded_association_columns,
    )

    print(f"Analyzed {len(frame):,} rows and {len(frame.columns):,} columns.")
    print(f"Wrote tables and visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
