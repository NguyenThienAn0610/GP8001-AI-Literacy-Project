"""Audit saved nested-search results and write concise comparison CSVs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluation import neural_scores, validation_key


def source_matches(metadata):
    """Historical results remain auditable when the working code has changed."""
    recorded = metadata.get('source_sha256', {
        'benchmark_deep_learning.py': metadata.get('script_sha256'),
    })
    source_dir = Path(__file__).parent
    return all(
        (source_dir / name).is_file()
        and hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == digest
        for name, digest in recorded.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('reports/deep_learning_search'))
    parser.add_argument('--data', type=Path, help='Override the recorded dataset path.')
    parser.add_argument('--output-dir', type=Path, help='Write audit outputs here; defaults to report-dir.')
    parser.add_argument('--strict-source', action='store_true', help='Also require unchanged training source files.')
    args = parser.parse_args()
    out = args.report_dir
    metadata = json.loads((out / 'experiment.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'complete'
    data = args.data or Path(metadata.get('data_path', 'data/raw/Career_Path_Dataset_with_Career.csv'))
    assert hashlib.sha256(data.read_bytes()).hexdigest() == metadata['dataset_sha256']
    matching_source = source_matches(metadata)
    if args.strict_source and not matching_source:
        raise ValueError('Training source differs from the recorded version.')
    trials = json.loads((out / 'trials.json').read_text(encoding='utf-8'))
    folds = pd.read_csv(out / 'fold_metrics.csv')
    summary = pd.read_csv(out / 'model_comparison.csv')
    assignments = pd.read_csv(out / 'split_assignments.csv')
    configurations = {c['id']: c for c in metadata['configs']}
    row_count, class_count = metadata['rows'], len(metadata['labels'])
    fold_numbers = sorted(assignments['fold'].unique())
    assert len(trials) == metadata['total_search_trials'] == len(fold_numbers) * len(configurations)
    assert len(folds) == metadata['outer_test_results'] == len(fold_numbers) * len(summary)
    assert set((t['fold'], t['config_id']) for t in trials) == {
        (fold, config) for fold in fold_numbers for config in configurations}
    assert not folds.duplicated(['fold', 'model']).any()
    for fold, rows in assignments.groupby('fold'):
        assert len(rows) == row_count
        assert set(rows['row_index']) == set(range(row_count))
        counts = rows['role'].value_counts()
        assert set(counts.index) == {'inner_train', 'inner_validation', 'outer_test'}
        assert counts['inner_validation'] == int(np.ceil(
            0.2 * (counts['inner_train'] + counts['inner_validation'])))
    tests = assignments[assignments.role == 'outer_test']
    assert tests['row_index'].nunique() == len(tests) == row_count

    selected = []
    for row in folds.itertuples():
        if row.model in ('deep_mlp', 'residual_network', 'field_transformer', 'inner_selected_network'):
            candidates = [t for t in trials if t['fold'] == row.fold
                          and (t['family'] == row.model or row.model == 'inner_selected_network')]
            winner = max(candidates, key=lambda trial: validation_key(trial['validation']))
            assert row.selected_config == winner['config_id']
            cfg = configurations[row.selected_config]
            if row.model != 'inner_selected_network':
                assert row.refit_epochs == winner['best_epoch']
                assert row.parameters == winner['trainable_parameters']
                selected.append({'fold': row.fold, **cfg,
                                 'epochs': winner['best_epoch'], 'parameters': int(row.parameters),
                                 'inner_validation_accuracy': winner['validation']['accuracy'],
                                 'outer_accuracy': row.accuracy, 'outer_macro_f1': row.macro_f1})

    oof = np.load(out / 'out_of_fold_probabilities.npz')
    y, fold_ids = oof['actual'], oof['fold']
    assert len(y) == len(fold_ids) == row_count
    assert set(fold_ids) == set(fold_numbers)
    np.testing.assert_array_equal(fold_ids, tests.sort_values('row_index')['fold'].to_numpy())
    differences = []
    baseline_correct = oof['logistic_regression'].argmax(axis=1) == y
    for name in summary.model:
        probabilities = oof[name]
        assert probabilities.shape == (row_count, class_count)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        for fold in fold_numbers:
            mask = fold_ids == fold
            p, truth = probabilities[mask], y[mask]
            row = folds[(folds.model == name) & (folds.fold == fold)].iloc[0]
            computed = neural_scores(truth, p)
            for metric, value in computed.items():
                tolerance = 1e-6 if metric == 'log_loss' else 1e-12
                assert np.isclose(row[metric], value, atol=tolerance, rtol=0), (name, fold, metric)
        rows = folds[folds.model == name].sort_values('fold')
        reference = folds[folds.model == 'logistic_regression'].sort_values('fold')
        delta = rows.accuracy.to_numpy() - reference.accuracy.to_numpy()
        correct = probabilities.argmax(axis=1) == y
        differences.append({'model': name, 'accuracy_difference_pp': float(delta.mean()*100),
                            'folds_better': int((delta > 1e-12).sum()),
                            'folds_equal': int((np.abs(delta) <= 1e-12).sum()),
                            'folds_worse': int((delta < -1e-12).sum()),
                            'correct_only_this_model': int((correct & ~baseline_correct).sum()),
                            'correct_only_logistic_regression': int((~correct & baseline_correct).sum()),
                            'both_correct': int((correct & baseline_correct).sum()),
                            'both_wrong': int((~correct & ~baseline_correct).sum())})
        row = summary[summary.model == name].iloc[0]
        for metric in ('accuracy', 'macro_f1', 'top_3_accuracy', 'log_loss'):
            assert np.isclose(row[f'{metric}_mean'], rows[metric].mean(), atol=1e-12, rtol=0)
            assert np.isclose(row[f'{metric}_std'], rows[metric].std(ddof=0), atol=1e-12, rtol=0)
    hyperparameters = []
    for config_id, cfg in configurations.items():
        rows = [t for t in trials if t['config_id'] == config_id]
        assert len(rows) == len(fold_numbers)
        hyperparameters.append({**cfg,
            'validation_accuracy_mean': float(np.mean([t['validation']['accuracy'] for t in rows])),
            'validation_macro_f1_mean': float(np.mean([t['validation']['macro_f1'] for t in rows])),
            'selected_folds': sum(s['id'] == config_id for s in selected),
            'parameters_min': min(t['trainable_parameters'] for t in rows),
            'parameters_max': max(t['trainable_parameters'] for t in rows),
            'epochs_run_mean': float(np.mean([t['epochs_run'] for t in rows]))})
    validation = {'status': 'pass', 'search_trials': len(trials), 'fold_results': len(folds),
                  'outer_test_predictions_per_model': len(y), 'inner_outer_indices_disjoint': True,
                  'selection_uses_only_inner_validation': True,
                  'source_matches_recorded': matching_source,
                  'source_note': 'Source comparison is separate from saved-result integrity; use --strict-source to require both.',
                  'metrics_recomputed_from_saved_predictions': True,
                  'trials_reaching_epoch_limit': sum(t['hit_epoch_limit'] for t in trials),
                  'search_training_seconds': sum(t['seconds'] for t in trials),
                  'max_parameters_tested': max(t['trainable_parameters'] for t in trials),
                  'comparison_scope': 'Descriptive fold and paired-error comparisons, not a significance test'}
    # Only publish derived files after every integrity check succeeds.
    destination = args.output_dir or out
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(destination / 'selected_configurations.csv', index=False)
    pd.DataFrame(differences).to_csv(destination / 'paired_accuracy_differences.csv', index=False)
    pd.DataFrame(hyperparameters).to_csv(destination / 'hyperparameter_summary.csv', index=False)
    (destination / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps(validation, indent=2))
    print(summary[['model', 'accuracy_mean', 'macro_f1_mean', 'top_3_accuracy_mean']].to_string(index=False))
    print(pd.DataFrame(differences)[['model', 'accuracy_difference_pp', 'folds_better', 'folds_worse']].to_string(index=False))


if __name__ == '__main__':
    main()
