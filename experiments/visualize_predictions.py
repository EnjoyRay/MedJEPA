"""
Visualize Exp1 & Exp2: show input image transformations + model predictions.

Generates figures:
  - Summary figure (Exp1+Exp2 combined, CSV-only, no images needed)
  - Exp2 per-class analysis + dashboard (CSV-only)
  - Exp1 sample images with perturbations (requires raw data on server)
  - Exp2 sample images with occlusions (requires raw data + scores.npz on server)

Usage (offline mode — only summary + CSV-based figures):
    python visualize_predictions.py \
        --exp1_dir results/exp1_ijepa_full \
        --exp2_dir results/exp2_ijepa_full \
        --output_dir results/visualizations

Usage (full mode — requires dataset on server):
    python visualize_predictions.py \
        --exp1_dir results/exp1_ijepa_full \
        --exp2_dir results/exp2_ijepa_full \
        --data_dir /path/to/VinBigData_ChestXray \
        --output_dir results/visualizations
"""

import argparse
import csv
import os
import sys
import numpy as np

# Make experiments/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from shared.vindr_dataset import VINDR_CLASSES


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _prob_bar_color(prob, gt):
    """Green if prediction matches GT direction, red otherwise."""
    if gt > 0.5:
        return '#27ae60' if prob > 0.5 else '#c0392b'
    else:
        return '#27ae60' if prob <= 0.5 else '#e74c3c'


# ---------------------------------------------------------------------------
# Summary figure: Exp1 + Exp2 combined (CSV only)
# ---------------------------------------------------------------------------

def create_summary_figure(exp1_dir, exp2_dir, output_dir):
    """Single-page summary combining key results from both experiments."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 12))

    # === Exp1: Robustness curves ===
    rc_path = os.path.join(exp1_dir, 'robustness_curve.csv')
    ed_path = os.path.join(exp1_dir, 'embedding_drift.csv')

    if os.path.isfile(rc_path):
        with open(rc_path) as f:
            rows = list(csv.DictReader(f))
        groups = {}
        for r in rows:
            cond = r['condition']
            if '/' in cond:
                ptype = cond.split('/')[0]
                groups.setdefault(ptype, []).append(r)

        ax1 = fig.add_subplot(2, 2, 1)
        for ptype, items in groups.items():
            labels_x = ['clean'] + [i['condition'].split('/')[1] for i in items]
            aurocs = [float(rows[0]['macro_auroc'])] + [float(i['macro_auroc']) for i in items]
            ax1.plot(range(len(labels_x)), aurocs, 'o-', label=ptype, linewidth=1.5, markersize=4)
        ax1.set_title('Exp1: AUROC under Perturbations', fontweight='bold')
        ax1.set_ylabel('Macro AUROC')
        ax1.set_ylim(0.55, 0.85)
        ax1.legend(fontsize=7, loc='lower left')
        ax1.set_xticks(range(max(len(g) + 1 for g in groups.values())))
        ax1.grid(alpha=0.2)

    if os.path.isfile(ed_path):
        with open(ed_path) as f:
            drift_rows = list(csv.DictReader(f))
        ax2 = fig.add_subplot(2, 2, 2)
        drift_data = [(r['condition'], float(r['cosine_drift'])) for r in drift_rows]
        drift_data.sort(key=lambda x: x[1], reverse=True)
        conditions = [d[0].replace('/', '\n') for d in drift_data]
        values = [d[1] for d in drift_data]
        colors = ['#e74c3c' if v > 0.05 else '#f39c12' if v > 0.02 else '#27ae60' for v in values]
        ax2.barh(range(len(conditions)), values, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(conditions)))
        ax2.set_yticklabels(conditions, fontsize=6)
        ax2.set_title('Exp1: Embedding Cosine Drift', fontweight='bold')
        ax2.set_xlabel('Cosine Distance')

    # === Exp2: Sensitivity ===
    pd_path = os.path.join(exp2_dir, 'prediction_drop.csv')
    fd_path = os.path.join(exp2_dir, 'feature_drift.csv')

    pd_data, fd_data = [], []
    if os.path.isfile(pd_path):
        with open(pd_path) as f:
            pd_data = list(csv.DictReader(f))
    if os.path.isfile(fd_path):
        with open(fd_path) as f:
            fd_data = list(csv.DictReader(f))

    if pd_data and fd_data:
        ax3 = fig.add_subplot(2, 2, 3)
        conds = ['Lesion Occlusion', 'Control Occlusion']
        drops = [float(pd_data[1]['auroc_drop']), float(pd_data[2]['auroc_drop'])]
        bars3 = ax3.bar(conds, drops, color=['steelblue', 'coral'], alpha=0.85, width=0.5)
        for b, v in zip(bars3, drops):
            ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0003,
                     f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax3.set_title('Exp2: AUROC Drop (Lesion vs Control)', fontweight='bold')
        ax3.set_ylabel('AUROC Drop')
        ax3.axhline(y=0, color='gray', linewidth=0.5)

        ax4 = fig.add_subplot(2, 2, 4)
        cos_drifts = [float(fd_data[0]['cosine_drift']), float(fd_data[1]['cosine_drift'])]
        pred_drops = [float(pd_data[1]['mean_pred_drop_max_class']),
                      float(pd_data[2]['mean_pred_drop_max_class'])]
        x = np.arange(2); width = 0.35
        ax4.bar(x - width/2, cos_drifts, width, label='Cosine Drift',
                color=['steelblue', 'coral'], alpha=0.85)
        ax4.bar(x + width/2, pred_drops, width, label='Pred. Drop',
                color=['steelblue', 'coral'], alpha=0.45, hatch='//')
        ax4.set_xticks(x); ax4.set_xticklabels(conds)
        ax4.set_title('Exp2: Feature Drift & Prediction Drop', fontweight='bold')
        ax4.legend(fontsize=7)

    fig.suptitle('I-JEPA ViT-L/14: Experiment Summary\n'
                 '(VinBigData-ChestXray, frozen encoder + linear probe)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'summary_figure.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved summary figure -> {out_path}")


# ---------------------------------------------------------------------------
# Exp2 Dashboard: per-class analysis + interpretation (CSV only)
# ---------------------------------------------------------------------------

def create_exp2_dashboard(exp2_dir, output_dir):
    """Rich Exp2 dashboard from CSV files — no raw images needed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    pc_path = os.path.join(exp2_dir, 'prediction_drop_per_class.csv')
    pd_path = os.path.join(exp2_dir, 'prediction_drop.csv')
    fd_path = os.path.join(exp2_dir, 'feature_drift.csv')

    if not os.path.isfile(pc_path):
        print(f"[Exp2 Viz] No per-class CSV at {pc_path}")
        return

    with open(pc_path) as f:
        reader = list(csv.DictReader(f))
    classes = [r['class'] for r in reader]
    les_drops = [float(r['lesion_pred_drop']) for r in reader]
    ctrl_drops = [float(r['control_pred_drop']) for r in reader]

    pd_data, fd_data = [], []
    if os.path.isfile(pd_path):
        with open(pd_path) as f:
            pd_data = list(csv.DictReader(f))
    if os.path.isfile(fd_path):
        with open(fd_path) as f:
            fd_data = list(csv.DictReader(f))

    # --- Figure 1: Per-class comparison + difference ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    x = np.arange(len(classes)); width = 0.35
    ax.bar(x - width/2, les_drops, width, label='Lesion Occ', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, ctrl_drops, width, label='Control Occ', color='coral', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Prediction Drop')
    ax.set_title('Per-Class: How much does occlusion change predictions?\n'
                 '(Negative = prediction increased after occlusion)', fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.2)

    ax2 = axes[1]
    diff = np.array(les_drops) - np.array(ctrl_drops)
    colors_diff = ['#27ae60' if d > 0.005 else '#e74c3c' if d < -0.005 else '#95a5a6' for d in diff]
    ax2.bar(x, diff, color=colors_diff, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('(Lesion Drop) - (Control Drop)')
    ax2.set_title('Key Result: Does lesion occlusion hurt MORE than control?\n'
                  '(~0 means model ignores lesion location)', fontsize=10)
    ax2.axhline(y=0, color='black', linewidth=1); ax2.grid(axis='y', alpha=0.2)

    mean_diff = np.mean(diff)
    ax2.text(0.98, 0.95, f'Mean Delta = {mean_diff:.4f}\n(Virtually zero -> no lesion specificity)',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_per_class_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved exp2_per_class_analysis.png")

    # --- Figure 2: Full dashboard ---
    fig = plt.figure(figsize=(14, 10))

    # (a) AUROC comparison
    ax_a = fig.add_subplot(2, 2, 1)
    if pd_data:
        conds = ['Original', 'Lesion\nOccluded', 'Control\nOccluded']
        aurocs = [float(r['macro_auroc']) for r in pd_data]
        colors_a = ['#2c3e50', '#3498db', '#e67e22']
        bars = ax_a.bar(conds, aurocs, color=colors_a, alpha=0.85, width=0.55)
        for b, v in zip(bars, aurocs):
            ax_a.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                      f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax_a.set_ylabel('Macro AUROC'); ax_a.set_title('Classification Performance', fontweight='bold')
        ax_a.set_ylim(0.55, 0.70); ax_a.grid(axis='y', alpha=0.2)

    # (b) Feature drift
    ax_b = fig.add_subplot(2, 2, 2)
    if fd_data:
        cos_d = [float(r['cosine_drift']) for r in fd_data]
        l2_d = [float(r['l2_drift']) for r in fd_data]
        x2 = np.arange(2); w = 0.35
        ax_b.bar(x2 - w/2, cos_d, w, label='Cosine Drift', color=['#3498db','#e67e22'], alpha=0.85)
        ax_b.bar(x2 + w/2, l2_d, w, label='L2 Drift', color=['#3498db','#e67e22'], alpha=0.4, hatch='//')
        ax_b.set_xticks(x2); ax_b.set_xticklabels(['Lesion Occ','Control Occ'])
        ax_b.set_title('Embedding Drift After Occlusion', fontweight='bold')
        ax_b.legend(fontsize=8); ax_b.grid(axis='y', alpha=0.2)

    # (c) Classes ranked by impact
    ax_c = fig.add_subplot(2, 2, 3)
    sort_idx = np.argsort(np.abs(les_drops))[::-1]
    cs_sorted = [classes[i] for i in sort_idx]
    ls_sorted = [les_drops[i] for i in sort_idx]
    ct_sorted = [ctrl_drops[i] for i in sort_idx]
    xs = np.arange(len(cs_sorted))
    ax_c.barh(xs - 0.18, ls_sorted, 0.35, label='Lesion', color='steelblue', alpha=0.85)
    ax_c.barh(xs + 0.18, ct_sorted, 0.35, label='Control', color='coral', alpha=0.85)
    ax_c.set_yticks(xs); ax_c.set_yticklabels(cs_sorted, fontsize=7)
    ax_c.set_xlabel('Prediction Drop'); ax_c.set_title('Classes Ranked by Impact of Occlusion', fontweight='bold')
    ax_c.axvline(x=0, color='black', linewidth=0.5)
    ax_c.legend(fontsize=8); ax_c.invert_yaxis()

    # (d) Interpretation text box
    ax_d = fig.add_subplot(2, 2, 4); ax_d.axis('off')

    v_ld = float(pd_data[1]['auroc_drop']) if len(pd_data) > 1 else 0
    v_cd = float(pd_data[2]['auroc_drop']) if len(pd_data) > 2 else 0
    v_lc = float(fd_data[0]['cosine_drift']) if fd_data else 0
    v_cc = float(fd_data[1]['cosine_drift']) if len(fd_data) > 1 else 0

    ax_d.text(0.05, 0.95, f"""
    EXP2 KEY FINDINGS
    ================================

    Q2: Does I-JEPA encode lesion regions?

    ANSWER: NO -- Evidence is strong.

    - Lesion AUROC drop:   {v_ld:.4f}
    - Control AUROC drop:  {v_cd:.4f}
    - Difference:          {v_ld - v_cd:.4f}  (~noise level)

    - Lesion cosine drift: {v_lc:.4f}
    - Control cosine drift:{v_cc:.4f}

    INTERPRETATION:
    Occluding the annotated lesion region does NOT
    cause more representation/prediction change than
    occluding a random control region.

    => I-JEPA's ViT-L/14 encoder does NOT specifically
       attend to pathological regions.
    => The learned representations are global/statistical,
       not localized to diagnostic findings.

    This supports the "boundary behavior" hypothesis:
    I-JEPA's invariance extends to BOTH nuisance factors
    AND diagnostically relevant features.
    """, transform=ax_d.transAxes, fontsize=9, verticalalignment='top',
              fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

    fig.suptitle('I-JEPA Exp2: Lesion Region Sensitivity -- Full Analysis Dashboard\n'
                 '(ViT-L/14, VinBigData-ChestXray, 879 images with bboxes)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_dashboard.png'), dpi=160, bbox_inches='tight')
    plt.close()
    print("Saved exp2_dashboard.png")


# ---------------------------------------------------------------------------
# Exp1 sample visualization (requires raw images — run on server)
# ---------------------------------------------------------------------------

def visualize_exp1(data_dir, exp1_dir, output_dir, n_samples=6,
                   img_size=224, in_chans=1):
    """Show original + perturbed images with GT labels. Requires raw dataset."""
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from shared.vindr_dataset import VinDrCXRDataset
    from shared.perturbations import get_perturbation_schedule

    os.makedirs(output_dir, exist_ok=True)
    dataset = VinDrCXRDataset(data_dir=data_dir, split='test', img_size=img_size, in_chans=in_chans)

    rng = np.random.RandomState(42)
    has_finding, no_finding = [], []
    for idx in range(len(dataset)):
        lab = dataset[idx]['label'].numpy()
        n_pos = int(lab[:-1].sum())
        if n_pos >= 3:
            has_finding.append(idx)
        elif lab[-1] > 0.5:
            no_finding.append(idx)

    selected = rng.choice(has_finding, size=min(n_samples*2//3, len(has_finding)), replace=False).tolist()
    rem = n_samples - len(selected)
    if rem > 0 and no_finding:
        selected.extend(rng.choice(no_finding, size=min(rem, len(no_finding)), replace=False).tolist())
    while len(selected) < n_samples:
        idx = rng.randint(0, len(dataset))
        if idx not in selected:
            selected.append(idx)

    pert_keys = ['gaussian_noise/sigma=0.30', 'gaussian_blur/sigma=3.0',
                 'contrast/factor=0.25', 'brightness/delta=+0.2']
    schedule = get_perturbation_schedule()
    pert_fns = {}
    for key in pert_keys:
        ptype, plabel = key.split('/', 1)
        for lbl, fn in schedule.get(ptype, []):
            if lbl == plabel:
                pert_fns[key] = fn; break

    print(f"[Exp1 Viz] {len(selected)} samples, {len(pert_keys)} perturbations")
    for si, sidx in enumerate(selected):
        sample = dataset[sidx]
        img_01 = (sample['image'].numpy() * 0.5 + 0.5).clip(0, 1)
        label = sample['label'].numpy(); img_id = sample['image_id']

        def to_disp(arr):
            d = np.repeat(arr[0], 3, axis=0) if arr.shape[0]==1 else np.transpose(arr,(1,2,0))
            return np.repeat(d[...,None] if d.ndim==2 else d, 3, axis=-1)[...,:3] if d.shape[-1]==1 else d

        imgs = [('Original', to_disp(img_01))]
        for key in pert_keys:
            if key not in pert_fns: continue
            t = torch.from_numpy(img_01).unsqueeze(0)
            p = pert_fns[key](t).squeeze(0).numpy()
            imgs.append((key.split('/')[-1], to_disp(p)))

        n_cols = len(imgs)
        fig = plt.figure(figsize=(n_cols * 2.8, 7))
        gs = GridSpec(2, n_cols, height_ratios=[2.5, 1], hspace=0.08, wspace=0.15)
        for col, (title, im) in enumerate(imgs):
            ax_img = fig.add_subplot(gs[0, col])
            ax_img.imshow(np.transpose(im, (1,2,0)) if im.shape[0]==3 or im.ndim==3 else im,
                          cmap='gray', vmin=0, vmax=1)
            ax_img.set_title(title, fontsize=9, fontweight='bold'); ax_img.axis('off')

            ax_bar = fig.add_subplot(gs[1, col])
            y_pos = np.arange(len(VINDR_CLASSES))
            colors = ['#27ae60' if l > 0.5 else '#bdc3c7' for l in label]
            ax_bar.barh(y_pos, label, color=colors, height=0.6, alpha=0.8)
            ax_bar.set_xlim(0, 1.05); ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(VINDR_CLASSES, fontsize=6) if col==0 else ax_bar.set_yticklabels([])
            ax_bar.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_bar.tick_params(labelsize=5)
            if col==0: ax_bar.set_xlabel('GT Label', fontsize=7)

        findings = [VINDR_CLASSES[i] for i in range(14) if label[i]>0.5]
        fig.suptitle(f'Exp1 Sample #{si+1} (id={img_id})\nFindings: {findings if findings else ["None"]}',
                     fontsize=10, y=1.02)
        plt.savefig(os.path.join(output_dir, f'exp1_sample_{si+1:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved exp1_sample_{si+1:02d}.png")
    print(f"[Exp1 Viz] Done")


# ---------------------------------------------------------------------------
# Exp2 sample visualization (requires raw images + scores.npz — run on server)
# ---------------------------------------------------------------------------

def visualize_exp2(data_dir, exp2_dir, output_dir, n_samples=8, img_size=224, in_chans=1):
    """Show original / lesion-occ / control-occ with predictions. Needs raw data + scores.npz."""
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    from shared.vindr_dataset import VinDrCXRDataset
    from experiments.exp2_lesion_sensitivity.occlusion import occlude_lesion, occlude_control

    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(exp2_dir, 'scores.npz')
    if not os.path.isfile(scores_path):
        print(f"[Exp2 Viz] No scores.npz at {scores_path}"); return

    scores = np.load(scores_path)
    orig_scores = scores['original_scores']; les_scores = scores['lesion_scores']
    ctrl_scores = scores['control_scores']; labels = scores['labels']

    dataset = VinDrCXRDataset(data_dir=data_dir, split='test', img_size=img_size, in_chans=in_chans)

    # Build mapping: dataset idx -> score idx (only for images with boxes)
    didx_to_sidx = {}; si = 0
    for di in range(len(dataset)):
        if len(dataset[di]['boxes']) > 0:
            if si < len(orig_scores): didx_to_sidx[di] = si
            si += 1

    # Pick interesting samples
    interest = np.array([np.abs(orig_scores[i]-les_scores[i]).max()*(1+int(labels[i][:14].sum()))
                         for i in range(min(len(orig_scores), len(didx_to_sidx)))])
    top = np.argsort(interest)[-max(n_samples*2//3,1):][::-1]
    sel = list(top[:n_samples*2//3])
    rest = set(range(len(orig_scores))) - set(sel)
    if n_samples - len(sel) > 0 and rest:
        sel.extend(np.random.RandomState(42).choice(list(rest), min(n_samples-len(sel), len(rest)), replace=False).tolist())
    sel = sel[:n_samples]

    print(f"[Exp2 Viz] {len(sel)} samples")
    for si, sidx in enumerate(sel):
        didx = next((d for d,s in didx_to_sidx.items() if s==sidx), None)
        if didx is None: continue

        sample = dataset[didx]; boxes = sample['boxes']; gt = sample['label'].numpy()
        img_01 = (sample['image'].numpy()*0.5+0.5).clip(0,1); img_t = torch.from_numpy(img_01)
        les_01 = occlude_lesion(img_t, boxes).numpy(); ctrl_01 = occlude_control(img_t, boxes).numpy()

        def td(a): return (np.repeat(a[0],3,axis=0)[None] if a.shape[0]==1 else np.transpose(a,(1,2,0))[None])

        conds = [('Original', td(img_01), orig_scores[sidx]),
                 ('Lesion Occ', td(les_01), les_scores[sidx]),
                 ('Control Occ', td(ctrl_01), ctrl_scores[sidx])]

        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 3, height_ratios=[2.5, 1.2, 0.8], hspace=0.12, wspace=0.15)
        for col, (title, im_d, pred) in enumerate(conds):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(np.transpose(im_d[0],(1,2,0)), cmap='gray', vmin=0, vmax=1)
            if title=='Original' and boxes:
                for b in boxes:
                    ax.add_patch(Rectangle((b['x_min'],b['y_min']), b['x_max']-b['x_min'],
                                           b['y_max']-b['y_min'], lw=2, ec='#e74c3c', fill=False))
            ax.set_title(title, fontsize=10, fontweight='bold'); ax.axis('off')

            axb = fig.add_subplot(gs[1, col])
            yp = np.arange(len(VINDR_CLASSES))
            axb.barh(yp, pred, color=[_prob_bar_color(p,g) for p,g in zip(pred,gt)], height=0.65, alpha=0.85)
            axb.set_xlim(0,1.02); axb.set_yticks(yp)
            axb.set_yticklabels(VINDR_CLASSES,fontsize=6) if col==0 else axb.set_yticklabels([])
            axb.axvline(0.5,color='gray',ls='--',lw=0.5,alpha=0.6); axb.tick_params(labelsize=5.5)
            if col==0: axb.set_xlabel('Prediction Prob.',fontsize=7)

            if col>0:
                ad = fig.add_subplot(gs[2,col])
                df = orig_scores[sidx]-pred
                ad.barh(yp,df,color=['#c0392b'if d>.02 else'#27ae60'if d<-.02 else'#95a5a6'for d in df],
                        height=0.65,alpha=0.75)
                ma=max(abs(df.min()),abs(df.max()),0.05)
                ad.set_xlim(-ma*1.3,ma*1.3);ad.axvline(0,color='k',lw=0.5)
                ad.set_yticks(yp);ad.set_yticklabels([]);ad.tick_params(labelsize=5)
                ad.set_xlabel(f'Delta vs Orig ({title.split()[0]})',fontsize=7)
            else:
                agt = fig.add_subplot(gs[2,col])
                agt.barh(yp,gt,color=['#27ae60'if g>.5 else'#bdc3c7'for g in gt],height=0.65,alpha=0.4)
                agt.set_xlim(0,1.02);agt.set_yticks(yp);agt.set_yticklabels([])
                agt.tick_params(labelsize=5);agt.set_xlabel('Ground Truth',fontsize=7)

        findings = [VINDR_CLASSES[i] for i in range(14) if gt[i]>0.5]
        fig.suptitle(f'Exp2 #{si+1} | Findings: {findings or ["None"]}\n'
                     f'MaxConf: Orig={orig_scores[sidx].max():.3f} Les={les_scores[sidx].max():.3f} Ctl={ctrl_scores[sidx].max():.3f}',
                     fontsize=9,y=1.01)
        plt.savefig(os.path.join(output_dir,f'exp2_sample_{si+1:02d}.png'),dpi=150,bbox_inches='tight')
        plt.close()
        print(f"  Saved exp2_sample_{si+1:02d}.png")
    print("[Exp2 Viz] Done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize Exp1 & Exp2 predictions')
    parser.add_argument('--exp1_dir', default=None)
    parser.add_argument('--exp2_dir', default=None)
    parser.add_argument('--data_dir', default=None,
                        help='Path to VinBigData-ChestXray root (optional)')
    parser.add_argument('--output_dir', default='results/visualizations')
    parser.add_argument('--n_exp1_samples', type=int, default=6)
    parser.add_argument('--n_exp2_samples', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--in_chans', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Always generate (CSV-only figures)
    if args.exp1_dir and args.exp2_dir:
        create_summary_figure(args.exp1_dir, args.exp2_dir, args.output_dir)
    if args.exp2_dir:
        create_exp2_dashboard(args.exp2_dir, args.output_dir)

    # Full visualizations (require raw data on server)
    if args.exp1_dir and args.data_dir:
        try:
            visualize_exp1(args.data_dir, args.exp1_dir, args.output_dir,
                           args.n_exp1_samples, args.img_size, args.in_chans)
        except FileNotFoundError as e:
            print(f"[Exp1 Viz] Skipped — data not found locally: {e}")
            print("  (Run on server with --data_dir pointing to VinBigData)")

    if args.exp2_dir and args.data_dir:
        try:
            visualize_exp2(args.data_dir, args.exp2_dir, args.output_dir,
                           args.n_exp2_samples, args.img_size, args.in_chans)
        except FileNotFoundError as e:
            print(f"[Exp2 Viz] Skipped — data not found locally: {e}")
            print("  (Run on server with --data_dir pointing to VinBigData)")

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
