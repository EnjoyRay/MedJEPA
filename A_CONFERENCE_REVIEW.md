# A 会标准模拟审稿意见汇总

> 被审稿件：`EXPERIMENT_REPORT.md`  
> 审稿日期：2026-05-03  
> 审稿标准：NeurIPS / ICLR / CVPR / MICCAI 顶级会议口径  
> 审稿人设置：三个独立 subagent，分别从机器学习自监督、医学影像临床 AI、实验设计与统计严谨性角度评审  

## 总体结论

三位审稿人一致认为：项目方向有价值，问题意识明确，图表和实验框架已经具备论文雏形；但按照 A 会标准，当前版本仍更接近一份有潜力的实验报告，而不是完整、稳健、可接收的顶会论文。

| Reviewer | 视角 | Overall Score | Confidence | Recommendation |
|---|---|---:|---:|---|
| Reviewer 1 | 机器学习 / 自监督学习 / 方法贡献 | 4 / 10 | 4 / 5 | Reject / Weak Reject |
| Reviewer 2 | 医学影像 / 临床 AI / 数据有效性 | 5 / 10 | 4 / 5 | Reject in current form |
| Reviewer 3 | 实验设计 / 统计严谨性 / 可复现性 | 4 / 10 | 4 / 5 | Weak Reject / Reject |

平均分约为 **4.3 / 10**。如果作为 A 会投稿，当前会被拒；如果作为一个月后答辩材料，基础已经可用，但必须补齐关键实验和收紧结论。

最核心的共识问题：

1. 当前评估不是 official test benchmark，也没有外部验证。
2. MAE-H/300 尚未完成，MAE baseline 仍不够公平。
3. 缺少 multi-seed、bootstrap confidence interval 和 per-class analysis。
4. baseline 太少，无法判断现象是否为 MAE vs I-JEPA 特有。
5. bbox occlusion 证据弱，不能支撑强 localization 或 clinical interpretability claim。
6. 数据集命名和协议需要更精确，VinBigData / VinDr-CXR 的来源、标签、bbox、split 必须讲清楚。
7. 当前贡献需要更明确地定位为“医学 SSL 表征边界评估”或“lesion-aware robustness benchmark”，否则 novelty 偏弱。

## Area-Chair 式总评

这篇工作提出了一个有意义的问题：I-JEPA 是否真的比 MAE 更适合胸片医学影像表征，尤其是在鲁棒性和局部病灶敏感性方面。当前实验显示 I-JEPA-H 在 clean AUROC 上优于 MAE-H/97，但在 Gaussian noise 下表现脆弱，且 bbox occlusion 只显示很弱的 lesion-specific signal。

审稿人认可以下优点：

- 不只报告 clean AUROC，而是进一步分析 robustness 和 lesion sensitivity。
- 修复了 train/test overlap，使用 fixed held-out split。
- MAE vs I-JEPA 的问题设置有价值。
- 图表和 case studies 有助于形成论文叙事。
- 结论相对克制，没有过度宣称 I-JEPA 全面优于 MAE。

但当前拒稿风险很高。主要原因不是写作，而是证据强度不够：评估集只是 internal validation，baseline 对比不够公平，统计可靠性不足，occlusion protocol 有混杂因素，且缺少机制分析和外部验证。

如果目标是 A 会论文，需要把工作从“比较几个模型的实验报告”升级成一个更完整的研究贡献，例如：

- 一个系统性的医学 SSL robustness + lesion sensitivity evaluation protocol；
- 一个明确的发现：global SSL representation 与 lesion-local evidence encoding 存在分离；
- 或者一个更强的 lesion-aware self-supervised learning benchmark。

当前最合理的论文主张应收紧为：

> Under a frozen-probe fixed held-out validation protocol, I-JEPA-H learns stronger global CXR representations than MAE-H/97, but its robustness is perturbation-specific and current bbox occlusion evidence does not establish strong lesion-local sensitivity.

## Reviewer 1：机器学习 / 自监督学习审稿意见

### Summary

本文比较 MAE 与 I-JEPA 在胸片自监督表征学习中的表现。作者使用 MIMIC-CXR 作为上游预训练数据，在 VinBigData / VinDr-CXR labeled subset 上进行 frozen encoder + linear probe 评估，并设计了两个额外分析：扰动鲁棒性和 bbox lesion/control occlusion。

当前结果显示 I-JEPA-H 在 clean AUROC 上优于 MAE-H/97，I-JEPA-H/201 表现最好。但 robustness 结果并非一致支持 I-JEPA，尤其 Gaussian noise 下 I-JEPA 表征漂移很大。bbox occlusion 结果也只显示很弱的 lesion-specific signal。文章主要结论是：I-JEPA 学到更强的全局胸片表征，但 embedding-space prediction 本身不足以保证局部病灶证据被稳定编码。

这是一个有价值的实证方向，但以顶会论文标准看，目前稿件仍更像实验报告，而不是成熟论文。主要问题是 novelty 不足、实验控制不够严格、claim 偏强、缺少关键 baseline 和统计验证。

### Strengths

1. 问题意识有价值。文章没有只停留在 clean AUROC，而是关注 robustness 和 lesion sensitivity。
2. MAE vs I-JEPA 的比较有实际意义，代表 pixel reconstruction 与 embedding prediction 两类 SSL objective。
3. 使用 bbox occlusion 区分 lesion 与 control，比只做 global classification 更进一步。
4. 作者对结论保持一定克制，没有简单宣称 I-JEPA 全面优于 MAE。
5. 图表和 case study 有助于理解实验现象。

### Major Weaknesses

1. **技术贡献不足。** 当前没有提出新的 SSL 方法、新的 pretraining objective、新理论分析，也没有建立足够系统的新 benchmark。主要贡献是比较 MAE 和 I-JEPA 在一个医学影像场景下的表现，novelty 偏弱。

2. **MAE 与 I-JEPA 对比不够公平。** 当前主比较是 MAE-H/97 vs I-JEPA-H/95 vs I-JEPA-H/201。I-JEPA-H/201 训练更久，而 MAE-H/97 不是最终 300 epoch 版本，因此 clean AUROC 差异不能严格归因于 SSL objective。

3. **缺少关键 baseline。** 只比较 MAE 和 I-JEPA 不足以支撑顶会级结论。至少需要 ImageNet supervised ViT、random init、DINO/DINOv2 或其他 SSL baseline，最好再加入医学影像 foundation model 或 CLIP-style medical model。

4. **评估集不是 official test benchmark。** 当前 12k/3k fixed held-out split 只能算 internal validation，不是官方 test，也不是跨数据集验证。

5. **统计可靠性不足。** Clean AUROC 没有多 seed，没有 bootstrap CI。I-JEPA-H/201 与 I-JEPA-H/95 差距只有 0.0057，稳定性未知。

6. **robustness 扰动的临床合理性不足。** Gaussian noise、blur、brightness、contrast 是常见扰动，但当前没有证明这些强度对应真实胸片采集或处理变化。

7. **bbox occlusion 不能支撑强 localization claim。** bbox 面积接近整图、遮挡 artifact、bbox 不完整、多标签对应关系不唯一、control 区域可能包含关键解剖结构，这些都会削弱结论。

8. **缺少机制分析。** 文章主要报告结果，但没有深入解释为什么 I-JEPA 对 brightness 较稳、对 Gaussian noise 极不稳。建议增加 patch-level token drift、frequency sensitivity、attention/saliency、retrieval 或 feature spectrum 分析。

### Required Revisions

1. 补齐 MAE-H/300，并尽量提供 matched compute 的 MAE vs I-JEPA 对比。
2. 增加 ImageNet supervised ViT、random encoder、DINO/DINOv2 或其他 SSL baseline。
3. 对 clean AUROC、robustness drop、occlusion delta 增加 CI；linear probe 至少 3 seeds。
4. 报告 per-class AUROC、per-class robustness drop、per-class bbox occlusion effect。
5. 加强 occlusion protocol：说明 control sampling，排除异常大 bbox，尝试 mean-fill/blur-fill/inpainting 等遮挡方式。
6. 增加机制分析，例如 frequency sensitivity、token-level drift、attention 与 bbox overlap。
7. 加入外部验证或官方 test；若没有，明确称为 held-out validation study。
8. 收紧 claim，不能声称 I-JEPA 更适合胸片，只能说在当前 frozen-probe setting 下 clean AUROC 更高。

### Score

**Overall Score：4 / 10**  
**Confidence：4 / 5**  
**Recommendation：Reject / Weak Reject**

## Reviewer 2：医学影像 / 临床 AI 审稿意见

### Summary

本文研究 MIMIC-CXR 预训练的 MAE-H 和 I-JEPA-H chest X-ray 表征，并在 fixed held-out VinBigData / VinDr-CXR split 上评估扰动鲁棒性和 bbox lesion sensitivity。当前结果显示 I-JEPA-H clean AUROC 更高，但鲁棒性是 perturbation-specific，bbox occlusion 只提供弱 lesion-specific evidence。

项目方向有潜力，但按照顶级医学 AI 或 A 会标准，当前证据仍不够。主要问题是 dataset validity、clinical claim calibration、weak localization evidence、缺少 official-test/external validation，以及没有充分处理 label/bbox uncertainty。

### Strengths

1. 不只报告 clean AUROC，而是评估 robustness 和 lesion sensitivity，更符合临床关注。
2. fixed held-out split 修复很重要，明确报告了 train=12,000、test=3,000、overlap=0。
3. MAE 与 I-JEPA 的比较有价值，且“global representation 不等于 lesion-local evidence”这一发现具有临床相关性。
4. perturbation visualization 和 bbox occlusion case studies 提高了可读性。
5. 报告在多个地方承认 bbox occlusion 不是 causal localization proof，表述较谨慎。

### Major Weaknesses

1. **评估不是 official test benchmark。** fixed held-out split 适合作为 internal validation，但不能等同于 VinDr-CXR / VinBigData official test benchmark。

2. **数据集命名和来源模糊。** 报告交替使用 VinBigData 和 VinDr-CXR。顶级医学影像论文必须明确数据来源、release version、图像数量、label set、bbox 来源、预处理、view filtering，以及标签来自 Kaggle 版本还是官方 VinDr-CXR。

3. **没有外部验证。** 只在一个 VinDr-style held-out split 上评估，不能支持强 medical robustness claim。建议增加 CheXpert、NIH ChestX-ray14、PadChest 或官方 VinDr test。

4. **bbox occlusion 不支持强 clinical interpretability。** I-JEPA-H/201 的 max prediction drop delta 只有 +0.0053，统计显著但临床意义很弱。MAE 甚至 control-dominant。最强可防守结论是负面的：当前 SSL 表征没有显示 robust lesion-specific evidence use。

5. **bbox reliability 没有充分处理。** 有 bbox area ratio = 1.000 的样本，说明存在 whole-image 或 near-whole-image box。这类样本中 lesion/control occlusion 定义不清，应排除或单独分析。

6. **occlusion 可能引入 OOD artifact。** 如果用 constant patch 遮挡，模型可能响应人工 mask，而不是病灶缺失。需要 mask-value controls、blur occlusion、mean-fill 或 inpainting-style occlusion。

7. **clinical relevance 仍不充分。** 当前实验仍是 representation/classification-level，没有 radiologist comparison、patient-level outcome、calibration、operating point analysis 或疾病组 failure analysis。

8. **统计报告不足。** Clean AUROC 缺少 CI 和 multi-seed variance。linear probe 可能受 seed、优化器、类别不平衡、阈值影响。

9. **pretraining protocol 描述不足。** 需要说明 MIMIC-CXR preprocessing、resolution、frontal/lateral filtering、patient-level 去重、epoch 定义、batch size、optimizer、initialization 等。

### Required Revisions

1. 将当前评估重新定位为 internal fixed held-out validation，而不是 official benchmark。
2. 增加精确的数据集章节：source、release、label schema、bbox schema、preprocessing、split construction、patient/image-level disjointness。
3. 给所有主 AUROC 和 AUROC drops 加 confidence intervals。
4. linear probe 至少跑 3 seeds，或说明为什么 deterministic single-seed 足够。
5. 尽量增加 external validation 或 official test evaluation。
6. 将 bbox 相关 claim 改为 annotation-region sensitivity 或 bbox occlusion sensitivity，避免 clinical localization 过度表达。
7. 排除或单独分析 very large bbox，尤其 area 接近 1.0 的样本。
8. 增加 occlusion-method sensitivity：constant mask vs blur vs mean-fill / inpainting-style masking。
9. 增加 per-class AUROC 和 clinically meaningful disease group discussion。
10. 明确 MAE/I-JEPA pretraining fairness：initialization、epochs、resolution、batch size、compute budget、data exposure。

### Score

**Overall Score：5 / 10**  
**Confidence：4 / 5**  
**Recommendation：Reject in current form**

## Reviewer 3：实验设计 / 统计严谨性审稿意见

### Summary

本文比较 MAE-H 与 I-JEPA-H 在胸片自监督表征中的 clean classification、扰动鲁棒性和 bbox occlusion lesion sensitivity。问题有价值，实验方向也比单纯 AUROC 更有论文潜力。但以 A 会标准看，当前版本仍是较好的课程/答辩实验报告，尚不足以支撑顶会投稿。主要问题不是写作，而是 evaluation protocol、统计可靠性和 baseline fairness 还不够扎实。

### Strengths

1. 研究问题明确，不只问 I-JEPA 是否提升 AUROC，还问是否鲁棒、是否保留局部病灶信息。
2. 修复 train/test overlap，使用 deterministic held-out split，这是必要且正确的。
3. Exp1 perturbation 和 Exp2 lesion/control occlusion 形成较完整证据链。
4. 报告没有过度声称 bbox occlusion 证明 localization，较严谨。
5. 可视化有帮助，尤其 perturbation examples、robustness matrix、paired effects 和 case studies。

### Major Weaknesses

1. **当前评估不是 official test benchmark，也没有外部验证。** 12k/3k split 只能算 internal validation。

2. **缺少 multi-seed。** Clean AUROC、F1、robustness drop 都是单次 linear probe 结果，没有 3-5 seeds，也没有 AUROC bootstrap CI。

3. **baseline fairness 不清楚。** MAE-H/97、I-JEPA-H/95、I-JEPA-H/201 的训练 epoch、预训练数据、初始化方式、batch size、effective updates、resolution、optimizer 和 compute 没有严格对齐。

4. **MAE-H/300 尚未完成。** 如果最终 MAE-H/300 改善明显，当前结论可能变化。

5. **Exp2 occlusion protocol 的 causal interpretation 偏弱。** bbox area=1.000 会使 lesion/control 对比失真；control occlusion 可能遮挡关键解剖区域；bbox 不一定覆盖完整诊断证据。

6. **统计检验存在 multiple comparison 和 effect size 问题。** p-value 没有说明校正方式，且一些显著结果效应量极小。

7. **缺少关键 ablation。** 无法判断结论来自 encoder、linear probe、resolution、扰动强度、occlusion mask 还是 control sampling。

### Required Revisions

1. 完成 MAE-H/300，并用相同 fixed split 重跑 Exp1/Exp2。
2. linear probe 至少跑 3 seeds，最好 5 seeds，报告 mean ± std。
3. 对 clean AUROC、AUROC drop、lesion-control delta 做 bootstrap 95% CI。
4. 固化并公开 split csv，报告 train/test 类别分布和 overlap=0 验证。
5. 加入强 baseline：random init、ImageNet supervised/original MAE、至少一个非 JEPA/MAE SSL baseline。
6. 过滤或单独分析异常大 bbox，例如 area > 0.5 或 area=1.0。
7. 对 control occlusion 做多次采样，报告均值和方差。
8. 补 per-class AUROC、per-class robustness drop、label prevalence。
9. 明确所有训练和评估超参数。
10. 若目标是 A 会，至少补一个外部验证集或 official test benchmark。

### Score

**Overall Score：4 / 10**  
**Confidence：4 / 5**  
**Recommendation：Weak Reject / Reject**

## 三位审稿人的共同问题清单

### P0：必须解决，否则 A 会基本无望

1. **完成 MAE-H/300 下游实验。**  
   当前 MAE-H/97 不是最终 baseline。必须在相同 fixed split 下重跑 Exp1/Exp2。

2. **补统计可靠性。**  
   至少 3 个 linear probe seeds；clean AUROC、robustness drop、lesion-control delta 做 bootstrap 95% CI。

3. **明确数据集和 split。**  
   统一 VinBigData / VinDr-CXR 命名，说明来源、release、label schema、bbox schema、preprocessing、是否 patient-level disjoint，保存并报告 split csv。

4. **收紧 claim。**  
   不能说 official benchmark，不能说 clinical localization，不能说 I-JEPA 更适合胸片。应改成 fixed held-out frozen-probe setting 下的 global representation finding。

5. **加强 baseline fairness。**  
   说明 MAE/I-JEPA 的预训练数据、初始化、resolution、batch size、epoch、effective updates 和 compute budget。

### P1：强烈建议，决定论文质量上限

1. **增加 per-class analysis。**  
   报告 per-class AUROC、per-class robustness drop、per-class occlusion effect、label prevalence。

2. **加强 occlusion protocol。**  
   过滤 bbox area > 0.5 或 area=1.0 的样本；control occlusion 多次采样；比较 constant mask、mean-fill、blur-fill。

3. **增加机制分析。**  
   至少做一个：frequency sensitivity、token-level drift、attention/saliency 与 bbox overlap、nearest-neighbor retrieval、feature spectrum。

4. **补 baseline。**  
   random init、ImageNet supervised ViT、original ImageNet MAE 或 DINO/DINOv2。若时间有限，至少补 random init + ImageNet supervised/original MAE。

### P2：有时间再做，但对 A 会很有帮助

1. 外部验证：CheXpert、NIH ChestX-ray14、MIMIC-CXR labeled subset 或 official VinDr test labels。
2. full fine-tuning baseline，检验 frozen linear probe 结论是否能推广。
3. 真实 domain shift 或 scanner/site robustness，而不只是 synthetic perturbation。
4. radiologist-grounded qualitative analysis 或 calibration / operating point analysis。

## 一个月答辩前的现实优先级

如果目标是一个月后答辩，而不是立即 A 会投稿，建议按下面顺序推进：

### 第 1 优先级：补 MAE-H/300

MAE-H/300 是目前最容易被问住的问题。因为现在报告中 I-JEPA-H/201 明显训练更久，MAE-H/97 还不是最终版本。MAE-H/300 完成后必须重跑：

- Exp1 robustness
- Exp2 occlusion
- paper figures
- `EXPERIMENT_REPORT.md` 中所有 MAE baseline 对比

### 第 2 优先级：加 CI 和多 seed

至少做：

- clean AUROC bootstrap CI
- Exp1 robustness AUROC drop CI
- Exp2 lesion-control delta CI
- linear probe 3 seeds，如果算力不够，至少对最终主表跑 3 seeds

这会显著提升答辩可信度。

### 第 3 优先级：per-class 和 label prevalence

医学影像答辩一定会问：哪些疾病有效，哪些疾病失败？因此必须补：

- 每类样本数
- 每类 AUROC
- 每类 perturbation drop
- 每类 occlusion delta

### 第 4 优先级：修 occlusion 分析

至少做三个低成本修复：

- 过滤 bbox area > 0.5 或单独做 subgroup
- control occlusion 多次采样
- 增加 mean-fill / blur-fill 对照

这样可以避免 reviewer 说 occlusion 全是 artifact。

### 第 5 优先级：补一个简单 baseline

如果没有时间跑 DINO/DINOv2，至少补：

- random frozen ViT baseline
- ImageNet supervised ViT 或 original MAE baseline

这样能证明医学预训练和 SSL 不是没有意义。

## 建议修改论文主张

### 当前容易被攻击的主张

> I-JEPA 更适合胸片医学影像。

这个说法太强。当前实验不能支撑，因为：

- 没有 official test / external validation；
- MAE-H/300 未完成；
- I-JEPA 对 Gaussian noise 很脆弱；
- bbox lesion-specific evidence 很弱；
- baseline 太少。

### 更安全的主张

> Under a fixed held-out frozen-probe evaluation, I-JEPA-H achieves stronger global CXR representations than MAE-H/97, but its robustness is perturbation-specific and bbox occlusion does not provide strong evidence of lesion-local sensitivity.

### 答辩中文表述

> 我们的实验不是要证明 I-JEPA 已经全面优于 MAE，而是要分析它在医学影像中的能力边界。结果显示，I-JEPA 在 clean 分类上更强，说明它确实学到了更好的全局胸片表征；但它对 Gaussian noise 很敏感，并且 bbox 病灶遮挡没有稳定地产生远大于对照遮挡的影响。因此，I-JEPA 的全局表征优势并不自动转化为局部病灶证据编码能力。这也是后续医学自监督学习需要解决的问题。

## 最终判定

当前版本：

- 作为 FYP / 硕士答辩材料：**有较好基础，但还需补 MAE-H/300、统计和 per-class。**
- 作为 A 会投稿：**Reject / Weak Reject。**
- 作为未来论文方向：**有潜力，关键在于把实验协议做严谨，把 claim 收紧，把 baseline 和统计补足。**

