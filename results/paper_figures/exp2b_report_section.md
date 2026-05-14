### Exp2b：类别对齐的病灶证据遮挡实验

旧版 Exp2 的分析单位是整张图：只要图像含有 bbox，就一次性遮挡所有 bbox，并观察 15 类整体 AUROC 和 max-class score 的变化。这个设计可以作为遮挡压力测试，但不适合作为强病灶证据实验，因为 bbox 类别、图像级多标签和最终评估类别没有严格对齐。

因此我们新增 Exp2b。其分析单位是 `(image_id, class_name)`：只有当该疾病类别为阳性、且存在同类别 bbox 时，才纳入样本。对于每个样本，只遮盖该类别对应的 bbox，并生成 5 个面积和形状匹配、且不覆盖任何已标注 bbox 的 control masks。主指标是 target-class logit drop 的配对差值：

`delta = [logit_c(original) - logit_c(lesion_occluded)] - mean_k[logit_c(original) - logit_c(control_k)]`

这个指标直接回答：遮盖 radiologist 标注的该疾病区域，是否比遮盖同等大小的非标注区域更能降低该疾病本身的预测分数。

![Fig. 9. Exp2b overall class-aligned lesion evidence test.](results/paper_figures/fig9_exp2b_class_aligned_overall.png)

| 模型 | n | Lesion logit drop | Control logit drop | Delta | 95% CI | p-value |
|---|---:|---:|---:|---:|---:|---:|
| I-JEPA-H/95 | 2873 | +0.1600 | -0.0304 | +0.1904 | [+0.1610, +0.2201] | 5e-05 |
| I-JEPA-H/201 | 2873 | +0.1512 | -0.0622 | +0.2134 | [+0.1858, +0.2408] | 5e-05 |
| MAE-H/97 | 2873 | -0.2743 | -0.3401 | +0.0658 | [+0.0530, +0.0784] | 5e-05 |

I-JEPA-H/95, I-JEPA-H/201 同时满足 lesion drop 为正、delta 为正，说明这些模型存在一定类别对齐的病灶区域敏感性；但仍需要结合 CI、p-value 和分疾病结果判断效应是否足够强。
MAE-H/97 的 lesion drop 为负，表示遮挡后目标 logit 反而升高；其正 delta 只能说明 lesion 与 control 的相对差异，不能解释为更强 localization。
在满足 lesion drop 为正的模型中，效应最大的模型是 I-JEPA-H/201，其 delta 为 +0.2134。这个值应解释为 target-class logit 层面的额外下降，而不是 AUROC 或准确率的直接变化。

![Fig. 10. Exp2b localized vs diffuse group effects.](results/paper_figures/fig10_exp2b_group_effects.png)

| 模型 | 疾病组 | n | Delta logit drop | 95% CI |
|---|---|---:|---:|---:|
| I-JEPA-H/95 | diffuse_or_global | 1983 | +0.1759 | [+0.1393, +0.2162] |
| I-JEPA-H/95 | localized | 890 | +0.2227 | [+0.1830, +0.2637] |
| I-JEPA-H/201 | diffuse_or_global | 1983 | +0.2260 | [+0.1905, +0.2645] |
| I-JEPA-H/201 | localized | 890 | +0.1852 | [+0.1462, +0.2252] |
| MAE-H/97 | diffuse_or_global | 1983 | +0.0414 | [+0.0252, +0.0584] |
| MAE-H/97 | localized | 890 | +0.1201 | [+0.1022, +0.1391] |

分组结果是 Exp2b 的关键，因为胸片疾病并不都对应局部小病灶。`localized` 组更接近传统意义上的局部证据遮挡；`diffuse_or_global` 组包含心影增大、胸腔积液、间质性改变等更依赖全局结构或大范围纹理的标签。如果 localized 组强于 diffuse/global 组，说明模型确实在部分局部疾病上使用了 bbox evidence；如果两组都很弱，则说明 frozen SSL 表征的分类信息更可能来自全局上下文、共现关系或 bbox 外区域。

![Fig. 11. Exp2b class heatmap.](results/paper_figures/fig11_exp2b_class_heatmap.png)

![Fig. 12. Exp2b area response.](results/paper_figures/fig12_exp2b_area_response.png)

面积分析用于排除一个重要混杂因素：遮挡越大，模型越容易变化，但这不一定是病灶特异性。如果 delta 随 bbox 面积单调增大，且 control 也同步增强，则说明实验主要反映 mask area sensitivity；如果在中小 bbox 的 localized 疾病上仍有正 delta，才更支持 lesion-specific evidence 的解释。

![Fig. 13. Exp2b qualitative class-aligned cases.](results/paper_figures/fig13_exp2b_case_sheet.png)

总体上，Exp2b 比旧 Exp2 更适合作为论文主实验，因为它把问题从“遮盖任意 bbox 是否影响整体分类”收紧为“遮盖某类疾病的标注区域是否影响该疾病本身”。因此无论结果是否强阳性，结论都会更清晰：强阳性支持 bbox-aware local evidence；弱阳性或阴性则说明高 clean AUROC 并不自动等价于局部病灶敏感性。
