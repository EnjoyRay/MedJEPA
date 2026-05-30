"""
Analyze and visualize class-aligned Exp2 outputs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Analyze Exp2b class-aligned outputs')
    parser.add_argument('--exp2_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--min_class_n', default=5, type=int)
    return parser.parse_args()


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _area_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [-1e-9, 0.05, 0.15, 0.35, 0.60, 1.0]
    labels = ['<=5%', '5-15%', '15-35%', '35-60%', '>60%']
    df['bbox_area_bin'] = pd.cut(df['bbox_area_frac'], bins=bins, labels=labels)
    return df


def _summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        if len(sub) == 0:
            continue
        rows.append(
            {
                'group': str(group),
                'n': len(sub),
                'bbox_area_frac_mean': sub['bbox_area_frac'].mean(),
                'target_logit_drop_mean': sub['target_logit_drop'].mean(),
                'control_logit_drop_mean': sub['control_logit_drop_mean'].mean(),
                'delta_logit_drop_mean': sub['delta_logit_drop'].mean(),
                'delta_logit_drop_std': sub['delta_logit_drop'].std(ddof=1),
                'target_prob_drop_mean': sub['target_prob_drop'].mean(),
                'control_prob_drop_mean': sub['control_prob_drop_mean'].mean(),
                'delta_prob_drop_mean': sub['delta_prob_drop'].mean(),
                'delta_cosine_drift_mean': sub['delta_cosine_drift'].mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot_forest(per_class: pd.DataFrame, output_dir: Path, model_name: str, min_class_n: int) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = per_class[per_class['n'] >= min_class_n].copy()
    if df.empty:
        df = per_class.copy()
    df = df.sort_values('delta_logit_drop_mean')
    y = np.arange(len(df))
    means = df['delta_logit_drop_mean'].to_numpy()
    lo = df['delta_logit_ci95_low'].to_numpy()
    hi = df['delta_logit_ci95_high'].to_numpy()
    xerr = np.vstack([means - lo, hi - means])

    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.38 * len(df) + 1.2)))
    colors = np.where(means >= 0, '#7C3AED', '#F97316')
    ax.barh(y, means, color=colors, alpha=0.35)
    ax.errorbar(means, y, xerr=xerr, fmt='o', color='#111827', capsize=3, linewidth=1.1)
    labels = [f"{r.group} (n={int(r.n)})" for r in df.itertuples()]
    ax.set_yticks(y, labels)
    ax.axvline(0, color='#111827', linewidth=1)
    ax.set_xlabel('Target-class logit drop delta (lesion - matched control)')
    ax.set_title(f'Class-aligned Exp2b per-class effects: {model_name}')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    path = output_dir / 'class_aligned_per_class_forest.png'
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_group(per_group: pd.DataFrame, output_dir: Path, model_name: str) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = ['localized', 'diffuse_or_global', 'other']
    df = per_group.set_index('group').reindex([g for g in order if g in set(per_group['group'])]).reset_index()
    if df.empty:
        df = per_group.copy()
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    axes[0].bar(x - 0.18, df['target_logit_drop_mean'], 0.36, label='Lesion', color='#7C3AED')
    axes[0].bar(x + 0.18, df['control_logit_drop_mean'], 0.36, label='Control', color='#F97316')
    axes[0].set_xticks(x, [f"{g}\n(n={int(n)})" for g, n in zip(df['group'], df['n'])])
    axes[0].set_ylabel('Target-class logit drop')
    axes[0].set_title('Lesion vs control effect')
    axes[0].legend(frameon=False)

    axes[1].bar(x, df['delta_logit_drop_mean'], color=['#7C3AED' if v >= 0 else '#F97316' for v in df['delta_logit_drop_mean']])
    axes[1].axhline(0, color='#111827', linewidth=1)
    axes[1].set_xticks(x, [f"{g}\n(n={int(n)})" for g, n in zip(df['group'], df['n'])])
    axes[1].set_ylabel('Delta logit drop')
    axes[1].set_title('Lesion-specific signal')

    fig.suptitle(f'Class-aligned Exp2b disease-group analysis: {model_name}')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / 'class_aligned_group_effects.png'
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_area(df: pd.DataFrame, area_summary: pd.DataFrame, output_dir: Path, model_name: str) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    colors = df['disease_group'].map({'localized': '#7C3AED', 'diffuse_or_global': '#0F766E'}).fillna('#6B7280')
    axes[0].scatter(df['bbox_area_frac'], df['delta_logit_drop'], s=16, c=colors, alpha=0.45, edgecolors='none')
    axes[0].axhline(0, color='#111827', linewidth=1)
    axes[0].set_xlabel('Bbox area fraction')
    axes[0].set_ylabel('Delta logit drop')
    axes[0].set_title('Sample-level area response')

    area_summary = area_summary.dropna(subset=['group'])
    x = np.arange(len(area_summary))
    axes[1].bar(x, area_summary['delta_logit_drop_mean'], color='#2563EB', alpha=0.75)
    axes[1].axhline(0, color='#111827', linewidth=1)
    axes[1].set_xticks(x, [f"{g}\n(n={int(n)})" for g, n in zip(area_summary['group'], area_summary['n'])])
    axes[1].set_ylabel('Mean delta logit drop')
    axes[1].set_title('Area bins')

    fig.suptitle(f'Class-aligned Exp2b area analysis: {model_name}')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / 'class_aligned_area_response.png'
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return path


def _fdr_correct(pvalues: np.ndarray, method: str = 'bh') -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values (adjusted p-values)."""
    from statsmodels.stats.multitest import multipletests
    if pvalues.size == 0:
        return pvalues
    mask = np.isfinite(pvalues)
    result = np.full_like(pvalues, np.nan, dtype=float)
    if mask.sum() > 0:
        _, qvals, _, _ = multipletests(pvalues[mask], method=f'fdr_{method}')
        result[mask] = qvals
    return result


def _write_case_table(df: pd.DataFrame, output_dir: Path) -> None:
    picks = []
    specs = [
        ('largest_lesion_delta', df.sort_values('delta_logit_drop', ascending=False)),
        ('largest_control_delta', df.sort_values('delta_logit_drop', ascending=True)),
        ('near_zero_delta', df.iloc[(df['delta_logit_drop'].abs()).argsort()]),
    ]
    for name, sub in specs:
        for _, row in sub.head(5).iterrows():
            out = row.to_dict()
            out['case_type'] = name
            picks.append(out)
    pd.DataFrame(picks).to_csv(output_dir / 'class_aligned_case_candidates.csv', index=False)


def main():
    args = get_args()
    exp2_dir = Path(args.exp2_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp2_dir / 'class_aligned_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or exp2_dir.name

    per_sample = _read_required(exp2_dir / 'class_aligned_per_sample.csv')
    per_class = _read_required(exp2_dir / 'class_aligned_summary_by_class.csv')
    per_group = _read_required(exp2_dir / 'class_aligned_summary_by_group.csv')
    overall = _read_required(exp2_dir / 'class_aligned_overall.csv')

    # FDR correction for per-class delta_logit p-values
    if 'delta_logit_pvalue' in per_class.columns:
        pvals = per_class['delta_logit_pvalue'].to_numpy(dtype=float)
        per_class['delta_logit_qvalue'] = _fdr_correct(pvals, method='bh')
        per_class.to_csv(output_dir / 'class_aligned_summary_by_class.csv', index=False)

    per_sample = _area_bins(per_sample)
    area_summary = _summary(per_sample, 'bbox_area_bin')
    area_summary.to_csv(output_dir / 'class_aligned_summary_by_area_bin.csv', index=False)
    per_sample.to_csv(output_dir / 'class_aligned_per_sample_with_bins.csv', index=False)
    _write_case_table(per_sample, output_dir)

    paths = [
        _plot_forest(per_class, output_dir, model_name, args.min_class_n),
        _plot_group(per_group, output_dir, model_name),
        _plot_area(per_sample, area_summary, output_dir, model_name),
    ]

    with open(output_dir / 'class_aligned_analysis_summary.md', 'w', encoding='utf-8') as f:
        row = overall.iloc[0]
        f.write(f"# Class-aligned Exp2b summary: {model_name}\n\n")
        f.write(f"- n: {int(row['n'])}\n")
        f.write(f"- Mean target logit drop: {row['target_logit_drop_mean']:.4f}\n")
        f.write(f"- Mean control logit drop: {row['control_logit_drop_mean']:.4f}\n")
        f.write(
            f"- Delta logit drop: {row['delta_logit_drop_mean']:.4f} "
            f"[{row['delta_logit_ci95_low']:.4f}, {row['delta_logit_ci95_high']:.4f}], "
            f"p={row['delta_logit_pvalue']:.4g}\n"
        )
        # Report FDR-corrected per-class results
        if 'delta_logit_qvalue' in per_class.columns:
            n_sig_fdr = int((per_class['delta_logit_qvalue'] < 0.05).sum())
            f.write(f"- Per-class with FDR q < 0.05: {n_sig_fdr}/{len(per_class)}\n")
            sig_classes = per_class[per_class['delta_logit_qvalue'] < 0.05]
            if len(sig_classes) > 0:
                for _, r in sig_classes.iterrows():
                    f.write(f"  - {r['group']}: q={r['delta_logit_qvalue']:.4f}\n")
        f.write(f"- Figures: {', '.join(p.name for p in paths)}\n")

    print(f"Class-aligned analysis written to {output_dir}")


if __name__ == '__main__':
    main()
