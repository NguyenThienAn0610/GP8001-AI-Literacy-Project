# test.py
# Full ML pipeline: clean -> feature engineering -> baselines -> strong models -> TabPFN -> stacking -> SHAP
# 支持读取 CSV / TSV / TXT / Excel / PDF 表格
# Dependencies: pandas numpy scikit-learn matplotlib
# Optional: xgboost lightgbm catboost optuna shap tabpfn pdfplumber

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Optional libraries ----------------
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import catboost as cb
except Exception:
    cb = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    optuna = None

try:
    import shap
except Exception:
    shap = None

try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ---------------- Global config ----------------
DATA_PATH = "GP8001.pdf"      # 如果改为 Excel，请改成 "GP8001.xlsx"
TARGET = "TargetRole"
RANDOM_STATE = 42

COLUMNS = [
    "Age", "Gender", "City", "Education", "Major", "Status", "Score",
    "Skills", "SoftSkills", "Languages", "Certifications", "Domain",
    "WorkMode", "Personality", "Experience", "Achievement", "OtherSkill",
    "WillingToRelocate", "TargetRole"
]


# ---------------- 1. Data loading and cleaning ----------------
def _clean_pdf_cell(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()


def _is_header_row(row):
    """判断 pdfplumber 提取的行是否为表头。"""
    if not row:
        return False
    first = [str(x).strip().lower() for x in row[:5] if x is not None]
    header_words = {"age", "gender", "city", "education", "major", "status", "score"}
    return len(set(first) & header_words) >= 2


def _read_pdf_tables(path):
    """使用 pdfplumber 读取 PDF 中的表格，合并为 DataFrame。"""
    if pdfplumber is None:
        raise ImportError(
            "读取 PDF 需要 pdfplumber。请先安装：pip install pdfplumber\n"
            "或者将 PDF 表格另存为 CSV/Excel 后，把 DATA_PATH 改为对应文件。"
        )

    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    clean = [_clean_pdf_cell(c) for c in row]
                    if all(v == "" for v in clean):
                        continue
                    if _is_header_row(clean):
                        continue

                    # 只保留有效数据行
                    if len(clean) >= len(COLUMNS):
                        rows.append(clean[:len(COLUMNS)])
                    elif len(clean) >= 7:
                        # 不完整行也保留，后面填 Unknown
                        row_padded = clean + [""] * (len(COLUMNS) - len(clean))
                        rows.append(row_padded[:len(COLUMNS)])

    if not rows:
        raise ValueError(
            "未能从 PDF 中提取到表格。请确认 PDF 表格有清晰边框，"
            "或先手动导出为 CSV/Excel。"
        )

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def load_data(path=DATA_PATH):
    """自动读取 CSV/TSV/Excel/PDF；若文件不存在则使用内置 demo 数据。"""
    if not os.path.exists(path):
        print(f"{path} not found. Using built-in demo data.")
        demo = []
        roles = ["Data Scientist", "Business Analyst", "Software Engineer"]
        for i in range(12):
            role = roles[i % 3]
            demo.append([
                20 + i % 10,
                "Male" if i % 2 == 0 else "Female",
                "Delhi",
                "B.Tech",
                "CSE",
                "Student",
                60 + i,
                "Python, SQL",
                "Teamwork",
                "English",
                "AWS",
                "Data Science",
                "Remote",
                "Analytical",
                "Internship",
                "None",
                "Programming",
                "Yes",
                role,
            ])
        df = pd.DataFrame(demo, columns=COLUMNS)
        return df

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext in [".txt", ".tsv"]:
        df = pd.read_csv(
            path, sep="\t", header=None, names=COLUMNS,
            encoding="utf-8", on_bad_lines="skip"
        )
    elif ext == ".pdf":
        df = _read_pdf_tables(path)
    else:
        # 默认尝试 CSV
        df = pd.read_csv(path)

    # 统一列名
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

    # 如果没有列名或列数匹配，则重命名
    if not set(COLUMNS).issubset(df.columns):
        if df.shape[1] == len(COLUMNS):
            df.columns = COLUMNS
        elif TARGET in df.columns:
            # 保留原列，后续只使用 TARGET 和存在的列
            pass
        else:
            raise ValueError(
                f"无法识别列名。期望 {len(COLUMNS)} 列：{COLUMNS}，"
                f"实际 {df.shape[1]} 列：{list(df.columns)}"
            )

    # 数值列转换
    for c in ["Age", "Score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def clean_data(df):
    df = df.copy()

    # 删除全空列、重复行
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()

    if TARGET not in df.columns:
        raise ValueError(f"目标列 {TARGET} 不存在。当前列：{list(df.columns)}")

    # 缺失率 > 90% 的列删除，但保留 TARGET
    missing_ratio = df.isna().mean()
    drop_cols = [c for c in df.columns if c != TARGET and missing_ratio[c] > 0.9]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 常量列删除，但保留 TARGET
    drop_cols = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 数值 -> 中位数，类别 -> Unknown
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    # IQR 裁剪异常值
    for c in num_cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df[c] = df[c].clip(lower, upper)

    # 合并稀有目标类，避免交叉验证报错
    vc = df[TARGET].value_counts()
    rare = vc[vc < 3].index
    df[TARGET] = df[TARGET].where(~df[TARGET].isin(rare), "Other")

    df.to_csv("cleaned.csv", index=False)
    return df


# ---------------- 2. Automatic feature engineering ----------------
def auto_feature_engineering(df, target_col):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不存在")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    multilabel_cols = [
        "Skills", "SoftSkills", "Languages", "Certifications",
        "Experience", "Achievement", "OtherSkill"
    ]
    multilabel_cols = [c for c in multilabel_cols if c in cat_cols]
    simple_cat_cols = [c for c in cat_cols if c not in multilabel_cols]

    parts = []

    # 数值特征
    if num_cols:
        parts.append(X[num_cols].copy())

    # 多标签列 -> binary features
    for c in multilabel_cols:
        mlb = MultiLabelBinarizer()
        vals = (
            X[c]
            .fillna("")
            .astype(str)
            .apply(lambda s: [
                v.strip()
                for v in re.split(r"[,/;]", s)
                if v.strip() and v.strip().lower() != "none"
            ])
        )
        arr = mlb.fit_transform(vals)
        ml_df = pd.DataFrame(
            arr,
            columns=[f"{c}__{v}" for v in mlb.classes_],
            index=X.index
        )
        # 删除低频标签
        keep = ml_df.sum(axis=0) >= 2
        ml_df = ml_df.loc[:, keep]
        if ml_df.shape[1] > 0:
            parts.append(ml_df)

    # 简单类别列 -> one-hot
    if simple_cat_cols:
        cat_dummies = pd.get_dummies(
            X[simple_cat_cols].fillna("Unknown"),
            columns=simple_cat_cols,
            dummy_na=False
        )
        parts.append(cat_dummies)

    if parts:
        X_feat = pd.concat(parts, axis=1)
    else:
        X_feat = pd.DataFrame(index=X.index)

    X_feat = X_feat.fillna(0)

    # 删除常量特征
    X_feat = X_feat.loc[:, X_feat.nunique() > 1]

    # 删除高度相关特征
    if X_feat.shape[1] > 1:
        corr = X_feat.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X_feat.drop(columns=to_drop, inplace=True, errors="ignore")

    X_feat = X_feat.astype(float)
    return X_feat, y


# ---------------- Evaluation helper ----------------
def evaluate_model_cv(model, X, y, cv=3):
    y_series = pd.Series(y)
    vc = y_series.value_counts()
    min_count = vc.min() if len(vc) else 0
    n_splits = min(cv, int(min_count)) if min_count >= 2 else 2

    if n_splits < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE,
            stratify=y if len(vc) > 1 and min_count >= 2 else None
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return np.array([f1_score(y_test, pred, average="macro")])

    cv_split = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    try:
        scores = cross_val_score(
            model, X, y,
            cv=cv_split,
            scoring="f1_macro",
            n_jobs=-1
        )
    except Exception as e:
        print(f"交叉验证失败，回退到 train_test_split: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores = np.array([f1_score(y_test, pred, average="macro")])

    return scores


# ---------------- 4. Optuna tuning ----------------
def tune_with_optuna(name, X, y, n_trials=3):
    if optuna is None:
        return {}

    def objective(trial):
        if name == "XGBoost" and xgb is not None:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = xgb.XGBClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss"
            )
        elif name == "LightGBM" and lgb is not None:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            }
            model = lgb.LGBMClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            )
        elif name == "CatBoost" and cb is not None:
            params = {
                "iterations": trial.suggest_int("iterations", 50, 200),
                "depth": trial.suggest_int("depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            }
            model = cb.CatBoostClassifier(
                **params,
                random_seed=RANDOM_STATE,
                verbose=0
            )
        else:
            return 0.0

        return evaluate_model_cv(model, X, y).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def create_strong_model(name, params):
    if name == "XGBoost" and xgb is not None:
        return xgb.XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
    if name == "LightGBM" and lgb is not None:
        return lgb.LGBMClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    if name == "CatBoost" and cb is not None:
        return cb.CatBoostClassifier(
            **params,
            random_seed=RANDOM_STATE,
            verbose=0
        )
    return None


# ---------------- 5. TabPFN ----------------
def try_tabpfn(X, y):
    if TabPFNClassifier is None:
        print("TabPFN not installed. Skipping.")
        return None

    try:
        X_use = X
        if X.shape[1] > 100:
            X_use = SelectKBest(f_classif, k=100).fit_transform(X, y)

        if X_use.shape[0] > 1000:
            X_use, _, y, _ = train_test_split(
                X_use, y,
                train_size=1000,
                stratify=y,
                random_state=RANDOM_STATE
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_use, y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE
        )

        try:
            clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        except TypeError:
            clf = TabPFNClassifier(device="cpu")

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        score = f1_score(y_test, pred, average="macro")
        print(f"TabPFN f1_macro: {score:.4f}")
        return score
    except Exception as e:
        print(f"TabPFN failed. Skipping: {e}")
        return None


# ---------------- 6. Stacking ----------------
def build_stacking(models_dict, X, y, cv=3):
    y_series = pd.Series(y)
    vc = y_series.value_counts()
    min_count = vc.min() if len(vc) else 0
    n_splits = min(cv, int(min_count)) if min_count >= 2 else 2

    if n_splits < 2:
        print("类别样本太少，跳过 stacking。")
        return None, None, None

    cv_split = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    meta_features = []
    used_models = []

    for name, model in models_dict.items():
        try:
            oof = cross_val_predict(
                model, X, y,
                cv=cv_split,
                method="predict_proba",
                n_jobs=-1
            )
            meta_features.append(oof)
            used_models.append(name)
            print(f"  {name} OOF done: {oof.shape}")
        except Exception as e:
            print(f"  {name} OOF failed. Skipping: {e}")

    if not meta_features:
        print("No OOF features available. Skipping stacking.")
        return None, None, None

    meta_X = np.hstack(meta_features)
    meta_model = LogisticRegression(max_iter=2000)

    stack_scores = evaluate_model_cv(meta_model, meta_X, y, cv=n_splits)
    return meta_model, meta_X, stack_scores


# ---------------- 7. SHAP and statistical evaluation ----------------
def explain_with_shap(model, X, y):
    if shap is None:
        print("SHAP not installed. Skipping explanation.")
        return

    try:
        X_sample = X.sample(min(100, len(X)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            vals = np.abs(shap_values[0]).mean(0)
        else:
            vals = np.abs(shap_values).mean(0)

        imp = pd.Series(vals, index=X.columns).sort_values(ascending=False)
        print("SHAP global top 10 features:")
        print(imp.head(10))
    except Exception as e:
        print(f"SHAP explanation failed: {e}")


def summarize_scores(scores_dict):
    print("Model evaluation results:")
    for name, scores in scores_dict.items():
        if scores is None:
            continue
        mean = np.mean(scores)
        ci = 1.96 * np.std(scores) / np.sqrt(len(scores))
        print(f"  {name}: f1_macro={mean:.4f} ± {ci:.4f}")

    # Simple paired t-test
    valid_names = [
        n for n, s in scores_dict.items()
        if s is not None and len(s) > 1
    ]
    if len(valid_names) >= 2:
        try:
            from scipy.stats import ttest_rel
            best_name = max(valid_names, key=lambda n: np.mean(scores_dict[n]))
            baseline_name = "LogisticRegression" if "LogisticRegression" in scores_dict else valid_names[0]
            if best_name != baseline_name and len(scores_dict[best_name]) == len(scores_dict[baseline_name]):
                stat, p = ttest_rel(
                    scores_dict[best_name],
                    scores_dict[baseline_name]
                )
                print(f"Paired t-test {best_name} vs {baseline_name}: p={p:.4f}")
        except Exception as e:
            print(f"Statistical test skipped: {e}")


# ---------------- Main workflow ----------------
def main():
    print("=" * 60)
    print("1. Load and clean the data")
    raw = load_data()
    print(f"Raw data shape: {raw.shape}")
    cleaned = clean_data(raw)
    print(f"Cleaned data shape: {cleaned.shape}. Saved to cleaned.csv")

    print("=" * 60)
    print("2. Automatic feature engineering")
    X, y = auto_feature_engineering(cleaned, TARGET)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Feature matrix X: {X.shape}, classes: {len(le.classes_)}")

    print("=" * 60)
    print("3. Train simple baselines")
    baseline_models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    baseline_scores = {}
    for name, model in baseline_models.items():
        scores = evaluate_model_cv(model, X, y_enc)
        baseline_scores[name] = scores
        print(f"{name}: f1_macro={scores.mean():.4f}")

    print("=" * 60)
    print("4. Train stronger models + Optuna tuning")
    strong_models = {}
    strong_scores = {}

    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        if (
            (name == "XGBoost" and xgb is None) or
            (name == "LightGBM" and lgb is None) or
            (name == "CatBoost" and cb is None)
        ):
            print(f"{name} not installed. Skipping.")
            continue

        print(f"Optuna tuning {name} ...")
        best_params = tune_with_optuna(name, X, y_enc, n_trials=3)
        model = create_strong_model(name, best_params)

        if model is None:
            continue

        scores = evaluate_model_cv(model, X, y_enc)
        strong_models[name] = model
        strong_scores[name] = scores
        print(f"{name}: f1_macro={scores.mean():.4f}, best_params={best_params}")

    print("=" * 60)
    print("5. Add tabular foundation models")
    tabpfn_score = try_tabpfn(X, y_enc)
    if tabpfn_score is not None:
        print(f"TabPFN reference score: {tabpfn_score:.4f}")

    print("=" * 60)
    print("6. Stacking")
    all_models = {**baseline_models, **strong_models}
    meta_model, meta_X, stack_scores = build_stacking(all_models, X, y_enc)
    if stack_scores is not None:
        print(f"Stacking: f1_macro={stack_scores.mean():.4f}")

    print("=" * 60)
    print("7. Explain and evaluate")
    all_scores = {**baseline_scores, **strong_scores}
    if stack_scores is not None:
        all_scores["Stacking"] = stack_scores

    summarize_scores(all_scores)

    # Pick one tree model for SHAP
    best_tree = None
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        if name in strong_models:
            best_tree = strong_models[name]
            break
    if best_tree is None:
        best_tree = baseline_models["RandomForest"]

    try:
        best_tree.fit(X, y_enc)
        explain_with_shap(best_tree, X, y_enc)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")

    print("=" * 60)
    print("All steps completed.")


if __name__ == "__main__":
    main()
