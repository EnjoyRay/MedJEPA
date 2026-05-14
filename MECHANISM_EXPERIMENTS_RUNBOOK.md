# Exp3-Exp5 Mechanism Experiments Runbook

新增实验用于把当前工作从现象比较推进到机制分析和轻量改进验证。

## Experiments

| 实验 | 入口 | 主要输出 |
|---|---|---|
| Exp3 frequency sensitivity | `experiments/exp3_frequency_sensitivity/run_exp3.py` | `frequency_sensitivity.csv` |
| Exp4 token drift / saliency alignment | `experiments/exp4_mechanism/run_exp4.py` | `mechanism_per_sample.csv`, `mechanism_summary_by_group.csv`, case figures |
| Exp5 lightweight mitigation | `experiments/exp5_lightweight_mitigation/run_exp5.py` | `mitigation_results.csv`, `robust_probe.pth` |

## A800 Submission

在 `node01` 登录 shell 中执行：

```bash
cd /home/jchwang/ray/JEPA
bash scripts/run_mechanism_experiments_a800.sh /home/jchwang/ray/data/VinBigData_ChestXray
```

可选参数：

```bash
PBS_NODE=node08 MAX_EXP4_SAMPLES=600 bash scripts/run_mechanism_experiments_a800.sh /home/jchwang/ray/data/VinBigData_ChestXray
```

快速 smoke run：

```bash
PBS_NODE=node08 MAX_EXP4_SAMPLES=50 MAX_EXP5_TRAIN_BATCHES=20 MAX_EXP5_EVAL_BATCHES=20 \
  bash scripts/run_mechanism_experiments_a800.sh /home/jchwang/ray/data/VinBigData_ChestXray
```

## Expected Paper Figures

| 图 | 生成脚本 | 文件 |
|---|---|---|
| Fig. 14 | `plot_exp3.py` | `results/paper_figures/fig14_exp3_frequency_sensitivity.png` |
| Fig. 15 | `plot_exp4.py` | `results/paper_figures/fig15_exp4_mechanism_alignment.png` |
| Fig. 16 | `plot_exp5.py` | `results/paper_figures/fig16_exp5_lightweight_mitigation.png` |

## Interpretation Targets

- Exp3 判断 I-JEPA 的 Gaussian noise 脆弱性是否来自高频纹理敏感或整体 token distribution shift。
- Exp4 判断 token drift 和 saliency 是否与 radiologist bbox 对齐。
- Exp5 判断简单 denoise 或 robust probe training 是否能缓解噪声脆弱性。
