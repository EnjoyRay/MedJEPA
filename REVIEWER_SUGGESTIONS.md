# I-JEPA 医学影像项目 — 审稿人视角的改进建议

> 生成日期: 2026-04-20
> 状态: 待实施
> 适用范围: FYP2 论文完善、后续实验规划

---

## 总体评价

当前实验已验证了 I-JEPA 在医学影像（胸部X光）中的两个核心行为：
1. 对干扰因素具有一定鲁棒性（模糊强、对比度弱）
2. **不特异性地编码病灶区域信息**（遮挡病灶 ≈ 遮挡随机区域）

这是一个有价值的发现，但作为一篇完整的研究工作，以下方面需要补充。

---

## P0 — 必须完成（决定论文能否成立）

### 1. 缺少 MAE baseline 对比

**问题**: 当前只有 I-JEPA 单模型结果。论文核心论点是 "I-JEPA 的边界行为"，但没有 baseline 无法说明这是 I-JEPA 特有还是所有 SSL 模型的共性。

**审稿人会问**:
- MAE 在同样实验下表现如何？MAE 是否也对病灶不敏感？
- 如果 MAE 也同样不敏感 → 这是 "ViT encoder 通用行为"，不是 "I-JEPA 特有行为"
- 只有 I-JEPA vs MAE 有显著差异时，结论才有意义

**解决方案**:
- 运行完整的 MAE Exp1 + Exp2 实验
- 代码已就绪 (`run_exp1.py --model mae`, `run_exp2.py --model mae`)
- 需要 MAE 预训练权重（MIMIC-CXR 上训练的）
- 形成并排对比表格和图表

**预期产出**: I-JEPA vs MAE 在所有指标上的对比表

### 2. 统计检验缺失

**问题**: Exp2 核心结论是 "lesion drop ≈ control drop"（差值仅 0.0008），但目前只报告了均值，没有统计显著性检验。

**需要补充**:
- **配对 t-test 或 Wilcoxon signed-rank test**: 检验 lesion drift 和 control drift 是否有显著差异
- **效应量 (Cohen's d)**: 量化差异的实际大小
- **Per-sample scatter plot**: x轴=lesion cosine drift, y轴=control cosine drift, 看样本是否落在 y=x 对角线附近
- **Bootstrap confidence interval**: 对 AUROC drop 差值做区间估计

**实现方式**: 新增一个 `statistical_analysis.py` 脚本，基于现有 CSV/npz 数据即可完成，无需重新跑模型。

---

## P1 — 强烈建议补充（大幅提升论文质量）

### 3. Attention / Saliency 可视化

**问题**: 当前用 occlusion 间接推断模型 "不看病灶"，但更直接的方式是可视化模型的注意力分布。

**方案**:
- **Attention Roll-out**: 对 ViT 的 self-attention 做加权求和，得到空间 attention map
- **Grad-CAM**: 基于探针预测的梯度生成热力图
- 展示几张具体图像：attention map 叠加在原图上，直观显示模型关注哪里

**如果 attention map 显示**:
- 均匀分布全图 → 支持 "全局特征" 结论
- 聚焦在非病灶区域 → 更强的证据
- 聚焦在病灶区域 → 与 occlusion 结果矛盾，需要解释

**工作量**: 中等。需要写一个 `visualize_attention.py`，利用 ViT 的 attention weights。

### 4. 探针 (Linear Probe) 性能分析

**问题**: Clean AUROC 仅 0.80 (Exp1) / 0.62 (Exp2)，F1 极低 (~0.11)。审稿人会质疑基线性能。

**需要分析**:
- F1 低的原因：多标签分类 + 固定 0.5 阈值导致。应报告 **Optimal F1** (按类别选最佳阈值)
- 报告 **per-class AUROC**（某些类可能很高，被平均拉低）
- 探针学习曲线：验证 50 epochs 是否收敛
- 与文献中同类设置的性能对比

**快速修复**: 在现有代码基础上增加 per-class 指标输出和 threshold analysis。

### 5. 定性案例展示 (Qualitative Case Study)

**问题**: 全是数字和曲线，缺少直观的图像案例。

**内容**:
- 选 6-8 张代表性图像（不同病灶类型、不同严重程度）
- 每张图展示：原始图像 → 加噪/模糊后 → 遮挡病灶后
- 配上模型对每张图的 15 类预测概率条形图
- 标注 GT 和 Top-3 预测

**作用**: 让读者直观感受 "图像明显变化但模型预测几乎不变"

**注意**: 这部分需要在服务器上运行（需要原始图像数据）。脚本 `visualize_predictions.py` 已支持此功能。

---

## P2 — 建议补充（提升泛化性和深度）

### 6. 第二个数据集验证

**问题**: 只在 VinBigData-ChestXray 上做了实验。

**建议选项** (按推荐顺序):
| 数据集 | 优势 | 工作量 |
|--------|------|--------|
| NIH ChestX-ray14 | 公开、112K 张、14 类与 VinBigData 重叠 | 中（需下载数据+适配代码）|
| CheXpert | 22万张、有不确定性标签 | 中 |
| MIMIC-CXR (测试集) | 与预训练数据同源 | 小（队友已有数据）|
| ImageNet 子集 | 区分 "医学特有" vs "JEPA通用行为" | 大 |

**最低要求**: 至少加一个额外的医学影像数据集，证明结论不是数据集特定的。

### 7. Ablation Study（消融实验）

**问题**: 没有分析哪些设计选择导致了观察到的行为。

**关键消融维度**:

#### 7a. Mask Ratio 影响
- I-JEPA 默认使用高 mask ratio（如 0.6-0.8）
- 高 mask ratio 强制模型从少量可见 patch 推断整体 → 可能导致学全局特征而非局部细节
- **实验**: 用不同 mask ratio 训练的权重做对比（如果队友能提供）

#### 7b. Patch Size 影响
- ViT-L/14 使用 14×14 patch（很大！224×224 图像只有 16×16=256 个 patch）
- 大 patch 本身就模糊了局部细节
- **实验**: 如果有 ViT-L/16 或 ViT-B/16 权重可对比

#### 7c. Pre-training Epochs
- 当前权重只训练了 ~15 epoch（可能未完全收敛）
- **确认**: 更多 epoch 的权重是否改变结论？

#### 7d. Predictor 结构
- I-JEPA 的 predictor depth 影响表征性质
- 更深的 predictor 可能恢复更多局部信息

### 8. Exp2 测试集分析

**问题**: 879 张有 bbox 的图像，样本量偏小且可能不均衡。

**需要报告**:
- 每类有多少张图有 bbox？（Cardiomegaly 可能占大头）
- Bbox 大小分布（小病灶 vs 大病灶的影响是否不同？）
- 单 bbox vs 多 bbox 图像的比例
- Bootstrap 重采样下的置信区间

---

## P3 — 锦上添花（有时间再做）

### 9. 与已有文献的对齐讨论

**写作角度**:
- JEPA 原论文 (Assran et al. 2023) 声称学到 "semantic representations"
- 你的结果实际上**挑战**了这个说法（至少在医学影像领域）
- 需要讨论：为什么在 ImageNet 上有效的目标函数在医学影像上产生 "过度不变性"？
- 可能原因：医学影像的语义信息高度局部化（病灶往往只占图像一小部分），而 JEPA 的抽象目标恰恰忽略局部细节

### 10. 正面叙事框架

当前叙事偏向负面："I-JEPA 忽略病灶 = 缺陷"。也可以正面表述：

**可选 Frame A（批评角度）**:
> "I-JEPA 的不变性过于激进，在医学影像中同时忽略了干扰和诊断信息，限制了其下游应用价值。"

**可选 Frame B（中性发现角度）**:
> "我们系统性地量化了 I-JEPA 表征对局部信息的敏感性，发现其具有高度的全局不变性。这一特性在某些场景下是优势（鲁棒性），在其他场景下是局限（需要局部精细特征）。"

**建议**: 根据 MAE baseline 结果选择 frame。如果 MAE 也类似 → Frame B（这是 ViT/SSL 的通用特性）；如果 MAE 更关注局部 → Frame A（I-JEPA 特有的过度不变性）。

### 11. Embedding 分析

**额外分析方向**:
- **t-SNE / UMAP**: 可视化 clean vs perturbed vs occluded embeddings 的分布
- ** probing classifiers of different capacities**: linear vs 2-layer MLP vs full fine-tuning，看表征本身是否有信息只是线性探针提取不出来
- **Layer-wise analysis**: 取 ViT 不同层的 embedding 做 probe，看哪层最/最不敏感

---

## 实施路线图

### 第一阶段（本周）— 补齐核心缺陷
- [ ] 运行 MAE baseline (Exp1 + Exp2)
- [ ] 编写统计检验脚本 (t-test, CI, scatter plot)
- [ ] 补充 per-class AUROC 和 optimal F1 分析

### 第二阶段（下周）— 强化证据
- [ ] Attention map 可视化
- [ ] 定性案例图（服务器端运行 visualize_predictions.py）
- [ ] Exp2 数据集分布分析

### 第三阶段（论文写作前）— 提升完整性
- [ ] 第二数据集验证（如可行）
- [ ] Ablation 实验（取决于可用资源）
- [ ] t-SNE / embedding 分析

### 第四阶段（写作）— 完善叙事
- [ ] 根据所有结果确定 Frame (A or B)
- [ ] 撰写 Related Work 和 Discussion
- [ ] 制作论文图表

---

## 快速参考：当前已有 vs 缺失的实验

| 已有 | 缺失 |
|------|------|
| I-JEPA Exp1 (4种扰动 × 4强度) | MAE Exp1 (同样配置) |
| I-JEPA Exp2 (879图 occlusion) | MAE Exp2 (同样配置) |
| Macro AUROC / F1 | Per-class AUROC / Optimal F1 |
| Cosine / L2 drift | 统计检验 (t-test, CI) |
| Robustness 曲线图 | Per-sample scatter plot |
| Sensitivity bar chart | Attention / Saliency maps |
| Summary figure | 定性案例图 (原图+变换+预测) |
| PROJECT_OVERVIEW.md | 本文档 (审稿人建议) |
