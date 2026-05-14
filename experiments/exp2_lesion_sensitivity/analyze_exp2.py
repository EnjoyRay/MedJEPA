"""
Offline analysis for Exp2 lesion sensitivity results.

This script consumes the files produced by run_exp2.py and adds the analyses
needed for the thesis:
  - paired lesion-vs-control tests with bootstrap confidence intervals
  - bbox-size subgroup analysis
  - disease-type subgroup analysis
  - scatter/bar plots for paper figures

Example:
    python experiments/exp2_lesion_sensitivity/analyze_exp2.py \
        --exp2_dir results/exp2_ijepa_vith_epoch97 \
        --output_dir results/exp2_ijepa_vith_epoch97/analysis
"""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


GLOBAL_FINDINGS = {
    'Cardiomegaly',
    'Aortic enlargement',
    'Pleural effusion',
}

LOCAL_FINDINGS = {
    'Nodule/Mass',
    'Calcification',
    'Pneumothorax',
    'Lung Opacity',
    'Other lesion',
    'Consolidation',
    'Infiltration',
    'Atelectasis',
    'ILD',
    'Pleural thickening',
    'Pulmonary fibrosis',
}


def get_args():
    parser = argparse.ArgumentParser(description='Analyze Exp2 lesion sensitivity outputs')
    parser.add_argument('--exp2_dir', required=True, help='Directory produced by run_exp2.py')
    parser.add_argument('--output_dir', default=None, help='Where to write analysis outputs')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _bootstrap_ci(values, n_boot=10000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _sign_flip_pvalue(diffs, n_perm=20000, seed=42):
    """Two-sided paired sign-flip permutation test for mean(diff)=0."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return math.nan
    observed = abs(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def _cohen_dz(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return math.nan
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else math.nan


def _paired_test_row(df, lesion_col, control_col, name, n_boot, seed):
    lesion = df[lesion_col].to_numpy(dtype=float)
    control = df[control_col].to_numpy(dtype=float)
    diff = lesion - control
    ci_lo, ci_hi = _bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    return {
        'metric': name,
        'n': int(np.isfinite(diff).sum()),
        'lesion_mean': float(np.nanmean(lesion)),
        'control_mean': float(np.nanmean(control)),
        'delta_mean_lesion_minus_control': float(np.nanmean(diff)),
        'delta_ci95_low': ci_lo,
        'delta_ci95_high': ci_hi,
        'cohen_dz': _cohen_dz(diff),
        'sign_flip_pvalue': _sign_flip_pvalue(diff, seed=seed),
    }


def _class_group(classes):
    names = {c.strip() for c in str(classes).split('|') if c.strip()}
    if names & GLOBAL_FINDINGS and not (names & LOCAL_FINDINGS):
        return 'global_only'
    if names & LOCAL_FINDINGS and not (names & GLOBAL_FINDINGS):
        return 'local_only'
    if names & GLOBAL_FINDINGS and names & LOCAL_FINDINGS:
        return 'mixed_global_local'
    return 'other'


def _add_subgroup_columns(df):
    if 'bbox_area_frac' in df.columns:
        values = df['bbox_area_frac'].astype(float)
        try:
            df['bbox_size_group'] = pd.qcut(
                values.rank(method='first'), q=3, labels=['small', 'medium', 'large']
            )
        except ValueError:
            df['bbox_size_group'] = 'all'
    if 'lesion_classes' in df.columns:
        df['disease_group'] = df['lesion_classes'].apply(_class_group)
    return df


def _summarize_group(df, group_col):
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        rows.append({
            'group': str(group),
            'n': len(sub),
            'bbox_area_frac_mean': float(sub['bbox_area_frac'].mean())
            if 'bbox_area_frac' in sub else math.nan,
            'delta_cosine_drift_mean': float(sub['delta_cosine_drift'].mean()),
            'delta_cosine_drift_ci_low': _bootstrap_ci(sub['delta_cosine_drift'])[0],
            'delta_cosine_drift_ci_high': _bootstrap_ci(sub['delta_cosine_drift'])[1],
            'delta_pred_drop_absmax_mean': float(sub['delta_pred_drop_absmax'].mean()),
            'delta_pred_drop_absmax_ci_low': _bootstrap_ci(sub['delta_pred_drop_absmax'])[0],
            'delta_pred_drop_absmax_ci_high': _bootstrap_ci(sub['delta_pred_drop_absmax'])[1],
        })
    return pd.DataFrame(rows)


def _plot_scatter(df, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for lesion_col, control_col, name in [
        ('lesion_cosine_drift', 'control_cosine_drift', 'cosine_drift'),
        ('lesion_pred_drop_absmax', 'control_pred_drop_absmax', 'pred_drop_absmax'),
    ]:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(df[control_col], df[lesion_col], s=12, alpha=0.5)
        mn = float(np.nanmin([df[control_col].min(), df[lesion_col].min()]))
        mx = float(np.nanmax([df[control_col].max(), df[lesion_col].max()]))
        ax.plot([mn, mx], [mn, mx], color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Control occlusion')
        ax.set_ylabel('Lesion occlusion')
        ax.set_title(f'Per-sample lesion vs control: {name}')
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f'scatter_{name}.png', dpi=180)
        plt.close(fig)


def _plot_group_bars(group_df, group_col, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if group_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(group_df))
    labels = group_df['group'].astype(str).tolist()

    axes[0].bar(x, group_df['delta_cosine_drift_mean'])
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_title('Delta cosine drift')
    axes[0].set_ylabel('Lesion - control')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha='right')

    axes[1].bar(x, group_df['delta_pred_drop_absmax_mean'])
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title('Delta prediction drop')
    axes[1].set_ylabel('Lesion - control')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha='right')

    fig.suptitle(group_col)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f'group_{group_col}.png', dpi=180)
    plt.close(fig)


def _legacy_summary(exp2_dir, output_dir):
    """Handle old result folders without per_sample_results.csv."""
    pred_path = Path(exp2_dir) / 'prediction_drop.csv'
    drift_path = Path(exp2_dir) / 'feature_drift.csv'
    per_class_path = Path(exp2_dir) / 'prediction_drop_per_class.csv'
    rows = []
    if pred_path.exists():
        rows.append('prediction_drop.csv')
        pd.read_csv(pred_path).to_csv(Path(output_dir) / 'legacy_prediction_drop_copy.csv', index=False)
    if drift_path.exists():
        rows.append('feature_drift.csv')
        pd.read_csv(drift_path).to_csv(Path(output_dir) / 'legacy_feature_drift_copy.csv', index=False)
    if per_class_path.exists():
        pc = pd.read_csv(per_class_path)
        pc['delta_lesion_minus_control'] = pc['lesion_pred_drop'] - pc['control_pred_drop']
        pc.to_csv(Path(output_dir) / 'per_class_delta.csv', index=False)
    with open(Path(output_dir) / 'README.txt', 'w', encoding='utf-8') as f:
        f.write(
            'per_sample_results.csv was not found, so paired tests and subgroup analysis '
            'could not be computed for this old run. Re-run Exp2 with the updated code '
            'to generate per-sample outputs.\n'
        )
        f.write('Files copied/analyzed: ' + ', '.join(rows) + '\n')


def main():
    args = get_args()
    exp2_dir = Path(args.exp2_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp2_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = exp2_dir / 'per_sample_results.csv'
    if not per_sample_path.exists():
        _legacy_summary(exp2_dir, output_dir)
        print(f"No per-sample file found. Wrote legacy summary to {output_dir}")
        return

    df = pd.read_csv(per_sample_path)
    df = _add_subgroup_columns(df)
    df.to_csv(output_dir / 'per_sample_results_with_groups.csv', index=False)

    tests = [
        _paired_test_row(
            df, 'lesion_cosine_drift', 'control_cosine_drift',
            'cosine_drift', args.bootstrap, args.seed
        ),
        _paired_test_row(
            df, 'lesion_l2_drift', 'control_l2_drift',
            'l2_drift', args.bootstrap, args.seed
        ),
        _paired_test_row(
            df, 'lesion_pred_drop_absmax', 'control_pred_drop_absmax',
            'prediction_drop_absmax', args.bootstrap, args.seed
        ),
        _paired_test_row(
            df, 'lesion_pred_drop_mean_signed', 'control_pred_drop_mean_signed',
            'prediction_drop_mean_signed', args.bootstrap, args.seed
        ),
    ]
    pd.DataFrame(tests).to_csv(output_dir / 'paired_tests.csv', index=False)

    for group_col in ['bbox_size_group', 'disease_group']:
        if group_col in df.columns:
            group_df = _summarize_group(df, group_col)
            group_df.to_csv(output_dir / f'{group_col}_summary.csv', index=False)
            _plot_group_bars(group_df, group_col, output_dir)

    _plot_scatter(df, output_dir)
    print(f"Analysis written to {output_dir}")


if __name__ == '__main__':
    main()
