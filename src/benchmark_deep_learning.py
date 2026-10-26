"""Nested evaluation of deep MLPs, residual networks, and field Transformers.

Run from the project root with the optional PyTorch dependency installed:
    .venv\\Scripts\\python.exe src/benchmark_deep_learning.py

For each outer fold, search only its training rows, using an inner stratified
80/20 split. Fit preprocessing on inner training rows. Select configurations
and epochs by inner validation accuracy, with macro-F1 and loss as tie-breaks.
Refit each family's winner on the full outer training fold, replaying the
learning-rate schedule through the selected epoch. Evaluate the outer test
fold once. Existing data, trained models, reports, and slides are not replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from experiment_models import make_experiment
from career_ml import FEATURE_COLUMNS, TARGET_COLUMN, load_training_data
from evaluation import neural_scores as scores, summarize_folds, validation_key
from neural_models import (
    FAMILIES, configurations, preprocessing, feature_groups, fit_trial, refit, probabilities,
)


def write_progress(out, metadata, trials, fold_rows):
    (out / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out / 'trials.json').write_text(json.dumps(trials, indent=2), encoding='utf-8')
    if trials:
        pd.DataFrame([{k: v for k, v in t.items() if k not in ('history', 'validation')}
                      | {f'validation_{k}': v for k, v in t['validation'].items()}
                      for t in trials]).to_csv(out / 'tuning_trials.csv', index=False)
    if fold_rows:
        frame = pd.DataFrame(fold_rows)
        frame.to_csv(out / 'fold_metrics.csv', index=False)
        summaries = []
        for name, rows in frame.groupby('model', sort=False):
            summary = {'model': name, 'completed_folds': len(rows)}
            summary.update(summarize_folds(rows.to_dict('records'),
                                           ('accuracy', 'macro_f1', 'top_3_accuracy', 'log_loss')))
            summaries.append(summary)
        pd.DataFrame(summaries).to_csv(out / 'model_comparison.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/Career_Path_Dataset_with_Career.csv'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports/deep_learning_search'))
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.threads) < 1:
        parser.error('max-epochs, patience, and threads must be positive')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    data = load_training_data(args.data)
    X = data[FEATURE_COLUMNS]
    labels, y = np.unique(data[TARGET_COLUMN].astype(str), return_inverse=True)
    configs = configurations()
    config_map = {c['id']: c for c in configs}
    if args.smoke_test:
        train, valid = train_test_split(np.arange(len(X)), train_size=1200, test_size=300,
                                        stratify=y, random_state=123)
        pre = preprocessing('binary')
        tx = torch.as_tensor(pre.fit_transform(X.iloc[train]), dtype=torch.float32, device=device)
        vx = torch.as_tensor(pre.transform(X.iloc[valid]), dtype=torch.float32, device=device)
        ty = torch.as_tensor(y[train], dtype=torch.long, device=device)
        for family in FAMILIES:
            cfg = next(c for c in configs if c['family'] == family)
            trial = fit_trial(cfg, (tx, vx, feature_groups(pre)), ty, y[valid], len(labels), device, 42, 2, 2)
            net = refit(cfg, trial, tx, ty, feature_groups(pre), len(labels), device, 42)
            assert probabilities(net, vx).shape == (300, len(labels))
            print(json.dumps({k: v for k, v in trial.items() if k != 'history'}), flush=True)
        return
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if (args.report_dir / 'experiment.json').exists():
        raise FileExistsError('Choose a new --report-dir to preserve existing experiment results.')
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    model_names = ['logistic_regression', 'original_mlp', *FAMILIES,
                   'inner_selected_network', 'three_family_ensemble']
    oof = {name: np.full((len(X), len(labels)), np.nan, dtype=np.float32) for name in model_names}
    fold_ids = np.zeros(len(X), dtype=int)
    split_rows = []
    metadata = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'data_path': str(args.data),
        'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'source_sha256': {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('benchmark_deep_learning.py', 'neural_models.py',
                         'experiment_models.py', 'career_ml.py', 'evaluation.py')
        },
        'rows': len(X), 'raw_inputs': FEATURE_COLUMNS, 'labels': labels.tolist(),
        'duplicate_feature_rows': int(X.duplicated().sum()),
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'native_threads': args.threads, 'deterministic_algorithms': True,
        'max_epochs': args.max_epochs, 'early_stopping_patience': args.patience,
        'configs': configs, 'optimizer': 'AdamW', 'gradient_clip_norm': 1.0,
        'outer_cv': 'StratifiedKFold(5, shuffle=True, random_state=42)',
        'inner_split': '80/20 stratified holdout within each outer training fold, seed=4200+fold',
        'selection': 'Highest inner validation Top-1, macro-F1 and negative log loss as tie-breaks',
        'preprocessing': 'Fit only inner training rows during search; refit on outer training rows for final fit',
        'refit': 'Train all outer training rows for inner-selected epochs, replaying selected learning rates',
        'ensemble': 'Predeclared equal average of the three inner-selected family probabilities',
        'scope': 'Nested search on an already explored dataset, not new external validation',
        'deployment_models_changed': False,
        'references': ['https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW',
                       'https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html',
                       'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html'],
    }
    trials, fold_rows = [], []
    with threadpool_limits(limits=args.threads):
        for fold, (outer_train, outer_test) in enumerate(folds, 1):
            fold_ids[outer_test] = fold
            inner_train, inner_valid = train_test_split(outer_train, test_size=0.2,
                                                        stratify=y[outer_train], random_state=4200 + fold)
            assert not set(inner_train) & set(inner_valid)
            assert not set(outer_test) & (set(inner_train) | set(inner_valid))
            for role, indices in [('inner_train', inner_train), ('inner_validation', inner_valid), ('outer_test', outer_test)]:
                split_rows.extend({'fold': fold, 'row_index': int(i), 'role': role} for i in indices)
            print(f'FOLD {fold}/5: {len(inner_train)} inner training, {len(inner_valid)} validation, {len(outer_test)} test', flush=True)
            matrices = {}
            for encoding in ('binary', 'tfidf'):
                pre = preprocessing(encoding)
                tx = torch.as_tensor(pre.fit_transform(X.iloc[inner_train]), dtype=torch.float32, device=device)
                vx = torch.as_tensor(pre.transform(X.iloc[inner_valid]), dtype=torch.float32, device=device)
                matrices[encoding] = (tx, vx, feature_groups(pre))
            ty = torch.as_tensor(y[inner_train], dtype=torch.long, device=device)
            current_trials = []
            for number, cfg in enumerate(configs, 1):
                trial = fit_trial(cfg, matrices[cfg['encoding']], ty, y[inner_valid], len(labels),
                                  device, 42 + fold, args.max_epochs, args.patience)
                trial['fold'] = fold
                trials.append(trial)
                current_trials.append(trial)
                print(f"  {number:02d}/18 {cfg['id']}: val={trial['validation']['accuracy']:.4f} "
                      f"best_epoch={trial['best_epoch']} run={trial['epochs_run']} seconds={trial['seconds']:.1f}", flush=True)
                write_progress(args.report_dir, metadata, trials, fold_rows)
            del matrices, ty
            winners = {family: max((t for t in current_trials if t['family'] == family),
                                    key=lambda t: validation_key(t['validation'])) for family in FAMILIES}
            overall = max(winners.values(), key=lambda t: validation_key(t['validation']))
            family_predictions = {}
            for family, trial in winners.items():
                cfg = config_map[trial['config_id']]
                started = time.perf_counter()
                pre = preprocessing(cfg['encoding'])
                tx = torch.as_tensor(pre.fit_transform(X.iloc[outer_train]), dtype=torch.float32, device=device)
                test_x = torch.as_tensor(pre.transform(X.iloc[outer_test]), dtype=torch.float32, device=device)
                ty = torch.as_tensor(y[outer_train], dtype=torch.long, device=device)
                net = refit(cfg, trial, tx, ty, feature_groups(pre), len(labels), device, 1042 + fold)
                p = probabilities(net, test_x)
                family_predictions[family] = p
                oof[family][outer_test] = p
                fold_rows.append({'fold': fold, 'model': family, 'selected_config': cfg['id'],
                                  'refit_epochs': trial['best_epoch'], 'parameters': sum(t.numel() for t in net.parameters()),
                                  'inner_validation_accuracy': trial['validation']['accuracy'],
                                  'fit_seconds': time.perf_counter() - started, **scores(y[outer_test], p)})
                del net, tx, test_x, ty
            for name, p in [
                ('inner_selected_network', family_predictions[overall['family']]),
                ('three_family_ensemble', np.mean(list(family_predictions.values()), axis=0)),
            ]:
                oof[name][outer_test] = p
                fold_rows.append({'fold': fold, 'model': name,
                                  'selected_config': overall['config_id'] if name == 'inner_selected_network' else 'equal_probability_average',
                                  **scores(y[outer_test], p)})
            for name, reference in [('logistic_regression', 'logistic_regression'), ('original_mlp', 'mlp_neural_network')]:
                pre, estimator, _ = make_experiment(reference)
                tx = pre.fit_transform(X.iloc[outer_train])
                vx = pre.transform(X.iloc[outer_test])
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    estimator.fit(tx, data[TARGET_COLUMN].iloc[outer_train])
                assert np.array_equal(estimator.classes_, labels)
                p = estimator.predict_proba(vx)
                oof[name][outer_test] = p
                fold_rows.append({'fold': fold, 'model': name, 'selected_config': 'existing_fixed_configuration',
                                  'fit_seconds': time.perf_counter() - started,
                                  'warnings': [str(w.message) for w in caught], **scores(y[outer_test], p)})
            write_progress(args.report_dir, metadata, trials, fold_rows)
            np.savez_compressed(args.report_dir / 'out_of_fold_probabilities.npz',
                                actual=y, labels=labels, fold=fold_ids, **oof)
            pd.DataFrame(split_rows).to_csv(args.report_dir / 'split_assignments.csv', index=False)
            print('OUTER TEST:', json.dumps({r['model']: round(r['accuracy'], 4)
                                            for r in fold_rows if r['fold'] == fold}), flush=True)
    assert all(np.isfinite(p).all() for p in oof.values())
    assert (fold_ids > 0).all()
    prediction_rows = []
    for name, p in oof.items():
        order = np.argsort(p, axis=1, kind='stable')[:, ::-1]
        for index in range(len(X)):
            prediction_rows.append({'row_index': index, 'fold': int(fold_ids[index]), 'model': name,
                                    'actual': labels[y[index]], 'prediction': labels[p[index].argmax()],
                                    'top_3': '; '.join(labels[order[index, :3]])})
    pd.DataFrame(prediction_rows).to_csv(args.report_dir / 'out_of_fold_predictions.csv', index=False)
    metadata.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                    total_search_trials=len(trials), outer_test_results=len(fold_rows))
    write_progress(args.report_dir, metadata, trials, fold_rows)
    print((args.report_dir / 'model_comparison.csv').read_text(), flush=True)


if __name__ == '__main__':
    main()
