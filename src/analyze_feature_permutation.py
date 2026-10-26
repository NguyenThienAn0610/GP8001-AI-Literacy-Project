r"""Measure held-out raw-field reliance, conditioning other fields on interest.

Run: .venv\Scripts\python.exe src/analyze_feature_permutation.py
All other fields are shuffled within each interest group, so the experiment
tests information used to distinguish careers within that group. The joint
shuffle preserves associations among the 17 non-interest fields. Interest
itself is shuffled globally. This is predictive, model-specific evidence,
not causal feature importance or a test of real-world career suitability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from career_ml import FEATURE_COLUMNS, TARGET_COLUMN, make_pipeline, load_training_data

INTEREST = 'Fields_of_Interest'
OTHER = [column for column in FEATURE_COLUMNS if column != INTEREST]


def scores(model, frame, truth):
    probability = model.predict_proba(frame)
    predicted = model.classes_[np.argmax(probability, axis=1)]
    ranked = model.classes_[np.argsort(probability, axis=1, kind='stable')[:, ::-1]]
    return {
        'top1_accuracy': float(np.mean(predicted == truth)),
        'macro_f1': float(f1_score(truth, predicted, average='macro', zero_division=0)),
        'top3_accuracy': float(np.mean((ranked[:, :3] == truth[:, None]).any(axis=1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/Career_Path_Dataset_with_Career.csv'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('repeats and threads must be positive')
    source = args.data
    data = load_training_data(source)
    X, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, base_rows, captured_warnings = [], [], []
    started = time.perf_counter()
    report = args.report_dir
    report.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        for fold, (train, test) in enumerate(cv.split(X, y), start=1):
            print(f'Fold {fold}/5: fit Logistic Regression', flush=True)
            model = make_pipeline('logistic_regression')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(X.iloc[train], y.iloc[train])
            captured_warnings.extend({'fold': fold, 'message': str(w.message)} for w in caught)
            heldout = X.iloc[test].reset_index(drop=True)
            truth = y.iloc[test].to_numpy()
            base = scores(model, heldout, truth)
            base_rows.append({'fold': fold, **base})
            groups = list(heldout.groupby(INTEREST, sort=True).indices.values())
            for repeat in range(args.repeats):
                rng = np.random.default_rng(42000 + fold * 100 + repeat)
                conditional = np.arange(len(heldout))
                for indices in groups:
                    conditional[indices] = rng.permutation(indices)
                global_order = rng.permutation(len(heldout))
                for field in FEATURE_COLUMNS + ['ALL_OTHER_FIELDS_JOINTLY']:
                    modified = heldout.copy()
                    columns = OTHER if field == 'ALL_OTHER_FIELDS_JOINTLY' else [field]
                    order = global_order if field == INTEREST else conditional
                    for column in columns:
                        modified[column] = heldout[column].to_numpy()[order]
                    result = scores(model, modified, truth)
                    row = {'fold': fold, 'repeat': repeat + 1, 'feature': field}
                    for key, value in result.items():
                        row[f'permuted_{key}'] = value
                        row[f'drop_{key}_pp'] = 100 * (base[key] - value)
                    rows.append(row)
            pd.DataFrame(rows).to_csv(report/'feature_permutation_repeats.csv', index=False)
            print(f'Fold {fold}/5 done, baseline Top-1={base["top1_accuracy"]:.4f}, '
                  f'elapsed={time.perf_counter()-started:.1f}s', flush=True)

    frame = pd.DataFrame(rows)
    keys = ['top1_accuracy', 'macro_f1', 'top3_accuracy']
    summary = []
    for field in FEATURE_COLUMNS + ['ALL_OTHER_FIELDS_JOINTLY']:
        values = frame[frame.feature == field]
        row = {'feature': field,
               'shuffle': 'global' if field == INTEREST else 'within_interest'}
        for key in keys:
            fold_means = values.groupby('fold')[f'drop_{key}_pp'].mean()
            row[f'mean_drop_{key}_pp'] = float(fold_means.mean())
            row[f'fold_std_drop_{key}_pp'] = float(fold_means.std(ddof=0))
            row[f'positive_folds_{key}'] = int((fold_means > 0).sum())
            row[f'mean_permuted_{key}'] = float(values[f'permuted_{key}'].mean())
        summary.append(row)
    summary_frame = pd.DataFrame(summary).sort_values('mean_drop_top1_accuracy_pp', ascending=False)
    summary_frame.to_csv(report/'feature_permutation_summary.csv', index=False)
    pd.DataFrame(base_rows).to_csv(report/'feature_permutation_baseline_folds.csv', index=False)
    metadata = {
        'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'sklearn_version': sklearn.__version__, 'rows': len(data),
        'model': 'Current make_pipeline(logistic_regression), including all 18 fields',
        'folds': 5, 'stratified': True, 'shuffle': True, 'seed': 42,
        'repeats_per_fold': args.repeats, 'native_threads': args.threads,
        'permutation': 'Raw fields before full fitted preprocessing; non-interest fields shuffled within interest. Joint condition uses one shared row permutation for all 17 fields.',
        'baseline_fold_means': {key: float(np.mean([r[key] for r in base_rows])) for key in keys},
        'importance_unit': 'Percentage points of score drop; macro-F1 differences also multiplied by100.',
        'uncertainty': 'Fold standard deviations describe variation, not confidence intervals. Repeats and folds are not independent inference samples.',
        'limits': 'Model-specific predictive relevance; no causal conclusion, no proof of feature independence, no claim that low-importance fields are irrelevant to real careers.',
        'seconds': time.perf_counter()-started,
        'warnings': captured_warnings,
        'summary': summary,
    }
    (report/'feature_permutation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(summary_frame[['feature','mean_drop_top1_accuracy_pp','fold_std_drop_top1_accuracy_pp',
                         'mean_drop_top3_accuracy_pp']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
