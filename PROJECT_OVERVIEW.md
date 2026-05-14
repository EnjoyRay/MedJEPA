# JEPA Medical Imaging — FYP2 Project Overview

## Research Topic

Analyzing I-JEPA's boundary behavior in medical imaging (chest X-rays). Specifically, we investigate whether the JEPA-style self-supervised learning objective leads to representations that are more "semantically meaningful" compared to MAE (Masked Autoencoder), using chest X-ray datasets.

**Baseline model**: MAE (Masked Autoencoder, He et al. 2022)  
**Target model**: I-JEPA (Image JEPA, Assran et al. 2023)

---

## Core Research Questions

### Q1 — Does I-JEPA truly ignore "things that should be ignored"?
Noise, texture artifacts, and distribution shifts (e.g., scanner variability) are nuisance factors that ideally should not affect semantic representations. We test whether I-JEPA's representations are more robust to these perturbations than MAE.

### Q2 — Does I-JEPA also ignore "things that shouldn't be ignored"?
Lesion regions in chest X-rays carry critical diagnostic information. We test whether I-JEPA's encoder actually encodes lesion-specific features or is insensitive to them.

---

## Datasets

| Role | Dataset | Notes |
|------|---------|-------|
| Pre-training (upstream) | MIMIC-CXR | Large chest X-ray corpus; used by partner to train I-JEPA |
| Downstream experiments | VinDr-CXR | Chest X-rays with lesion bounding box annotations |

---

## Experiment Design

### Exp1: Robustness to Nuisance Perturbations (`experiments/exp1_robustness/`)

**Idea**: Feed original and perturbed images through the frozen encoder, then compare downstream classification performance and embedding distances.

**Perturbation types** (3 categories):
1. Gaussian noise (σ = 0.1, 0.2, 0.3)
2. Gaussian blur (kernel σ = 1, 2, 3)
3. Brightness / contrast shift

**Metrics**:
- **Metric 1 — Functional change (Robustness Curve)**:  
  AUROC / F1 / accuracy at each perturbation strength. Lower drop = more robust.
- **Metric 2 — Representation drift**:  
  Cosine distance (or L2) between clean and perturbed embeddings.  
  If I-JEPA truly ignores nuisance, drift should be smaller than MAE.

**Pipeline**:
1. Load pre-trained encoder (I-JEPA or MAE)
2. Extract embeddings on clean images → train a linear probe on VinDr-CXR labels
3. Evaluate probe on perturbed test images → record AUROC/F1 per perturbation level
4. Also record cosine distance between clean and perturbed embeddings

**Split note**: if official `annotations/test.csv` is absent, the loader now creates a deterministic disjoint split from `annotations/train.csv`: 80% for probe training and 20% held out for evaluation. Older runs before this fix used a sampled test split but did not exclude those images from train, so final thesis numbers should be rerun with the fixed split.

### Exp2: Lesion Region Sensitivity (`experiments/exp2_lesion_sensitivity/`)

**Idea**: Use VinDr-CXR bounding boxes to occlude either the lesion region or an equivalent-sized control region, then measure how much the representation and prediction change.

**Occlusion strategy**:
- **Lesion mask**: zero out (or gray-fill) pixels inside the annotated bounding box
- **Control mask**: zero out an equivalent region outside the lesion (same size, different location)

**Metrics**:
- **Metric 1 — Prediction drop**:  
  How much does the classification score drop after occlusion?  
  Larger drop for lesion → model uses lesion features.
- **Metric 2 — Feature drift**:  
  Cosine distance between original and occluded embeddings.  
  Lesion occlusion should cause larger drift than control if model encodes lesion.

---

## Code Structure

```
JEPA/
├── PROJECT_OVERVIEW.md           ← this file
├── mae/                          ← MAE baseline (pre-existing code)
└── experiments/
    ├── shared/
    │   ├── model_wrappers.py     ← Unified interface for I-JEPA and MAE encoders
    │   ├── perturbations.py      ← Noise, blur, brightness transforms
    │   ├── vindr_dataset.py      ← VinDr-CXR dataset loader with bounding boxes
    │   └── metrics.py            ← Shared metric utilities (AUROC, cosine drift, etc.)
    ├── exp1_robustness/
    │   ├── run_exp1.py           ← Main entry point for Exp1
    │   ├── linear_probe.py       ← Linear probe training on top of frozen encoder
    │   └── evaluate_robustness.py← Robustness curve + embedding drift evaluation
    └── exp2_lesion_sensitivity/
        ├── run_exp2.py           ← Main entry point for Exp2
        ├── occlusion.py          ← Lesion/control occlusion logic
        └── evaluate_sensitivity.py ← Prediction drop + feature drift evaluation
```

---

## How to Run (once weights and data are ready)

### Exp1 — Robustness
```bash
# I-JEPA
python experiments/exp1_robustness/run_exp1.py \
    --model ijepa \
    --weights /path/to/ijepa_mimic.pth \
    --data_dir /path/to/vindr-cxr \
    --output_dir results/exp1_ijepa

# MAE
python experiments/exp1_robustness/run_exp1.py \
    --model mae \
    --weights /path/to/mae_mimic.pth \
    --data_dir /path/to/vindr-cxr \
    --output_dir results/exp1_mae
```

### Exp2 — Lesion Sensitivity
```bash
# I-JEPA
python experiments/exp2_lesion_sensitivity/run_exp2.py \
    --model ijepa \
    --weights /path/to/ijepa_mimic.pth \
    --data_dir /path/to/vindr-cxr \
    --output_dir results/exp2_ijepa

# MAE
python experiments/exp2_lesion_sensitivity/run_exp2.py \
    --model mae \
    --weights /path/to/mae_mimic.pth \
    --data_dir /path/to/vindr-cxr \
    --output_dir results/exp2_mae
```

---

## Experiment Results (Updated 2026-05-02)

### Paper Figures

Unified paper-style figures are generated by:

```bash
python scripts/build_paper_figures.py
```

Outputs:

- `results/paper_figures/fig0_perturbation_visual_examples.png`
- `results/paper_figures/fig1_exp1_robustness_matrix.png`
- `results/paper_figures/fig2_exp1_drop_drift_tradeoff.png`
- `results/paper_figures/fig3_exp2_occlusion_summary.png`
- `results/paper_figures/fig4_exp2_paired_effects.png`
- `results/paper_figures/fig5_exp2_per_sample_distribution.png`
- `results/paper_figures/fig6_exp2_subgroup_effects.png`
- `results/paper_figures/fig7_exp2_per_class_heatmap.png`
- `results/paper_figures/fig8_case_study_contact_sheet.png`

### Completed Runs

Note: the runs below were produced before the disjoint fallback split fix and should be rerun before final thesis reporting.

| Experiment | Model | Data | Epochs | Status | Results Location |
|-----------|-------|------|--------|--------|-----------------|
| Exp1 Full | I-JEPA ViT-L/14 | 15K train + 3K test | ~15 | Done (GPU 1) | `results/exp1_ijepa_full/` |
| Exp2 Full | I-JEPA ViT-L/14 | 879 images with bboxes | ~15 | Done (GPU 1) | `results/exp2_ijepa_full/` |
| Exp1 Full | I-JEPA ViT-H/14 | 15K train + 3K test | **97** | **Done (GPU 1)** | `results/exp1_ijepa_vith_epoch97/` |
| Exp2 Full | I-JEPA ViT-H/14 | 879 images with bboxes | **97** | **Done (GPU 1)** | `results/exp2_ijepa_vith_epoch97/` |
| Exp2 Per-sample | I-JEPA ViT-H/14 | 879 images with bboxes | **97** | **Done (UIC GPU 0)** | `results/exp2_ijepa_vith_epoch97_per_sample/` |
| Exp2 Per-sample | I-JEPA ViT-H/14 | 879 images with bboxes | **201** | **Done (UIC GPU 0)** | `results/exp2_ijepa_vith_ep201_per_sample/` |
| Exp1 Full | MAE ViT-H/14 | 15K train + 3K test | **97** | **Done (A800)** | `results/exp1_mae_huge_mimic_97ep/` |
| Exp2 Full | MAE ViT-H/14 | 879 images with bboxes | **97** | **Done (A800)** | `results/exp2_mae_huge_mimic_97ep/` |

### Exp1 Results - Robustness

#### Run 1: ViT-L/14 (~15 epochs)

**Clean**: AUROC=0.8037, F1=0.1113. Contrast was weakest point (factor=0.25 -> drift=0.185).

#### Run 2: ViT-H/14 (**97 epochs**) - NEW

**Clean**: AUROC=**0.9302**, F1=**0.4126** (+15.7%/+270% vs L/14!)

| Perturbation (strongest) | H/14 AUROC | H/14 Drop | Key Change vs L/14 |
|--------------------------|-----------|----------|-------------------|
| Noise sigma=0.30 | 0.6017 | -32.8% | **Now the #1 weakness!** (was moderate) |
| Blur sigma=3.0 | 0.7721 | -15.8% | Still robust at low sigma |
| Brightness delta=+0.2 | 0.8999 | -3.0% | **Much more robust** than L/14 |
| Contrast factor=0.25 | 0.8456 | -8.5% | **No longer weakest** (major improvement) |

#### Run 3: MAE ViT-H/14 (**97 epochs**) - BASELINE

**Clean**: AUROC=**0.9104**, F1=**0.2589**. MAE is slightly lower than I-JEPA-H on clean AUROC and substantially lower on F1, but it is more stable under mild Gaussian noise.

| Perturbation (selected) | MAE-H AUROC | MAE-H Drop | Key Comparison vs I-JEPA-H |
|--------------------------|------------|-----------|----------------------------|
| Noise sigma=0.05 | 0.8183 | -9.2% | **Much more robust** than I-JEPA-H at mild noise |
| Noise sigma=0.30 | 0.5103 | -40.0% | Both collapse; MAE becomes lower |
| Blur sigma=3.0 | 0.7541 | -15.6% | Similar to I-JEPA-H |
| Brightness delta=+0.2 | 0.8427 | -6.8% | Less robust than I-JEPA-H |
| Contrast factor=0.25 | 0.8086 | -10.2% | Less robust than I-JEPA-H |

### Exp2 Results - Lesion Sensitivity

#### Run 1: ViT-L/14 (~15 epochs)
Original AUROC=0.6234. Lesion drop=0.0090, Control drop=0.0098, Delta=**-0.0008**

#### Run 2: ViT-H/14 (**97 epochs, aligned per-sample**) - NEW
Original AUROC=**0.7568**. Lesion drop=0.0270, Control drop=0.0266, Delta=**+0.0004**

#### Run 3: ViT-H/14 (**201 epochs, aligned per-sample**) - NEW
Original AUROC=**0.8001**. Lesion drop=0.0154, Control drop=0.0124, Delta=**+0.0031**. Longer I-JEPA pretraining improves overall Exp2 AUROC, but lesion/control separation remains small.

#### Run 4: MAE ViT-H/14 (**97 epochs**) - BASELINE
Original AUROC=**0.7403**. Lesion drop=0.0135, Control drop=0.0150, Delta=**-0.0015**

**Conclusion after adding MAE and I-JEPA per-sample tests**: lesion occlusion effect remains close to control occlusion effect in I-JEPA and MAE. I-JEPA-201 shows a slight lesion advantage, but the effect is small; strong frozen SSL encoders can achieve useful global chest-X-ray classification performance without strong lesion-specific bbox sensitivity.

### Summary: Answers to Research Questions (Updated)

| Question | Answer | Evidence |
|----------|--------|----------|
| **Q1**: Does I-JEPA ignore nuisance factors? | **Selectively yes** | I-JEPA-H is strong for brightness/contrast/blur but weak to additive noise; MAE is better at mild noise |
| **Q2**: Does I-JEPA also ignore lesion regions? | **Yes, consistently** | Lesion ~= control in I-JEPA-L, I-JEPA-H, and MAE-H (Delta around 0) |
| **Q3**: Is I-JEPA different from MAE? | **Yes for robustness, not for lesion specificity** | I-JEPA-H has higher clean AUROC and better brightness/contrast robustness; both fail lesion-specific occlusion |

---

## Expected Output Files

### Exp1
- `results/exp1_*/robustness_curve.csv` — AUROC/F1 per perturbation type and strength
- `results/exp1_*/embedding_drift.csv` — mean cosine distance per perturbation type and strength
- `results/exp1_*/robustness_curve.png` — visualization

### Exp2
- `results/exp2_*/prediction_drop.csv` — prediction score before/after lesion vs. control occlusion
- `results/exp2_*/feature_drift.csv` — cosine distance for lesion vs. control occlusion
- `results/exp2_*/sensitivity_bar.png` — visualization

---

## VinDr-CXR Annotation Format

VinDr-CXR provides CSV annotations with columns:
`image_id, class_name, x_min, y_min, x_max, y_max, rad_id`

Images are in DICOM format (`.dicom`), converted to PNG/JPEG for training.  
Multi-label: each image can have multiple bounding boxes.

---

## Actual Output Files

### Exp1 — ViT-L/14 (`results/exp1_ijepa_full/`)
- `robustness_curve.csv`, `embedding_drift.csv`, `probe.pth`
- `gaussian_noise.png`, `gaussian_blur.png`, `brightness.png`, `contrast.png`

### Exp1 — ViT-H/14 epoch 97 (`results/exp1_ijepa_vith_epoch97/`) **NEW**
- `robustness_curve.csv`, `embedding_drift.csv`, `probe.pth`
- `gaussian_noise.png`, `gaussian_blur.png`, `brightness.png`, `contrast.png`

### Exp2 — ViT-L/14 (`results/exp2_ijepa_full/`)
- `prediction_drop.csv`, `prediction_drop_per_class.csv`, `feature_drift.csv`, `scores.npz`
- `sensitivity_bar.png`, `per_class_drop.png`

### Exp2 — ViT-H/14 epoch 97 (`results/exp2_ijepa_vith_epoch97/`) **NEW**
- `prediction_drop.csv`, `prediction_drop_per_class.csv`, `feature_drift.csv`, `scores.npz`
- `sensitivity_bar.png`, `per_class_drop.png`

### Exp1 — MAE ViT-H/14 epoch 97 (`results/exp1_mae_huge_mimic_97ep/`) **NEW**
- `robustness_curve.csv`, `embedding_drift.csv`, `probe.pth`, `args.json`
- `robustness_gaussian_noise.png`, `robustness_gaussian_blur.png`, `robustness_brightness.png`, `robustness_contrast.png`

### Exp2 — MAE ViT-H/14 epoch 97 (`results/exp2_mae_huge_mimic_97ep/`) **NEW**
- `prediction_drop.csv`, `prediction_drop_per_class.csv`, `feature_drift.csv`, `scores.npz`, `per_sample_results.csv`
- `sensitivity_bar.png`, `per_class_drop.png`
- `analysis/paired_tests.csv`, `analysis/bbox_size_group_summary.csv`, `analysis/disease_group_summary.csv`
- `case_studies/selected_cases.csv` and 8 qualitative case-study figures

---

## Notes

- Both experiments use a **frozen encoder** (no fine-tuning) to isolate the quality of pre-trained representations.
- Linear probe is trained on clean VinDr-CXR train split, evaluated on test split with perturbations.
- I-JEPA uses the **context encoder** (not the target encoder) for downstream tasks.
- For I-JEPA, we average the patch token embeddings (excluding CLS if not present) to get a single vector.
- For MAE, we use the CLS token or mean-pool patch tokens after the encoder.

## Environment & Runtime Notes (Updated 2026-05-02)

### Servers
- Original I-JEPA server: `ssh -p 6422 uic2@10.250.93.98`
- A800 server for MAE and downstream reruns: `ssh jchwang@10.169.143.54`
- A800 workspace: `/home/jchwang/ray/JEPA`
- A800 downstream data: `/home/jchwang/ray/data/VinBigData_ChestXray`
- MAE checkpoint: `/home/jchwang/ray/outputs/mae_huge_mimic_97ep_imagenet_init_256_bs32/checkpoint-96.pth`

### Code Adaptations Made
1. **model_wrappers.py**: Auto-detects MAE/I-JEPA architecture; supports ViT-H/14, ViT-L/14, ViT-B/16; handles PyTorch 2.6+ checkpoint loading with `weights_only=False`.
2. **vindr_dataset.py**: Supports both VinDr and VinBigData layouts; auto-falls back to train annotations for sampled evaluation when official test labels are unavailable.
3. **Exp2 analysis tools**: Added per-sample export, paired tests, subgroup analysis, scatter plots, and case-study figure generation.
4. **A800 run script**: Added `scripts/run_mae_downstream_a800.sh` for MAE Exp1/Exp2 downstream evaluation.

### Next Steps
- [x] Run I-JEPA ViT-H/14 epoch 97 experiments (Exp1 + Exp2)
- [x] Run MAE ViT-H/14 epoch 97 baseline experiments (Exp1 + Exp2)
- [x] Add MAE Exp2 paired tests, subgroup analysis, and qualitative case studies
- [x] Update experiment report with paper-level analysis
- [ ] Rerun I-JEPA-H Exp2 with the new per-sample export for symmetric paired tests
- [ ] Build final I-JEPA-H vs MAE-H combined figures for thesis
- [ ] Write thesis Results/Discussion sections from `EXPERIMENT_REPORT.md`
