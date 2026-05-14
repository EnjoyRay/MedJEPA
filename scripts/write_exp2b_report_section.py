"""Write a thesis-ready Exp2b report section from completed class-aligned outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_figures"

RUNS = {
    "I-JEPA-H/95": RESULTS / "exp2b_class_aligned_ijepa_vith_ep95_fixedsplit",
    "I-JEPA-H/201": RESULTS / "exp2b_class_aligned_ijepa_vith_ep201_fixedsplit",
    "MAE-H/97": RESULTS / "exp2b_class_aligned_mae_huge_mimic_97ep_fixedsplit",
    "MAE-H/300": RESULTS / "exp2b_class_aligned_mae_huge_mimic_300ep_fixedsplit",
}


def _fmt(value: float) -> str:
    return f"{float(value):+.4f}"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_path = OUT / "table_exp2b_class_aligned_overall.csv"
    group_path = OUT / "table_exp2b_class_aligned_groups.csv"
    class_path = OUT / "table_exp2b_class_aligned_classes.csv"
    if not overall_path.exists() or not group_path.exists() or not class_path.exists():
        raise FileNotFoundError("Run scripts/build_class_aligned_figures.py first.")
    return pd.read_csv(overall_path), pd.read_csv(group_path), pd.read_csv(class_path)


def build_markdown(overall: pd.DataFrame, groups: pd.DataFrame, classes: pd.DataFrame) -> str:
    lines = [
        "### Exp2b：类别对齐的病灶证据遮挡实验",
        "",
        "旧版 Exp2 的分析单位是整张图：只要图像含有 bbox，就一次性遮挡所有 bbox，并观察 15 类整体 AUROC 和 max-class score 的变化。这个设计可以作为遮挡压力测试，但不适合作为强病灶证据实验，因为 bbox 类别、图像级多标签和最终评估类别没有严格对齐。",
        "",
        "因此我们新增 Exp2b。其分析单位是 `(image_id, class_name)`：只有当该疾病类别为阳性、且存在同类别 bbox 时，才纳入样本。对于每个样本，只遮盖该类别对应的 bbox，并生成 5 个面积和形状匹配、且不覆盖任何已标注 bbox 的 control masks。主指标是 target-class logit drop 的配对差值：",
        "",
        "`delta = [logit_c(original) - logit_c(lesion_occluded)] - mean_k[logit_c(original) - logit_c(control_k)]`",
        "",
        "这个指标直接回答：遮盖 radiologist 标注的该疾病区域，是否比遮盖同等大小的非标注区域更能降低该疾病本身的预测分数。",
        "",
        "![Fig. 9. Exp2b overall class-aligned lesion evidence test.](results/paper_figures/fig9_exp2b_class_aligned_overall.png)",
        "",
        "| 模型 | n | Lesion logit drop | Control logit drop | Delta | 95% CI | p-value |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['n'])} | "
            f"{_fmt(row['target_logit_drop_mean'])} | {_fmt(row['control_logit_drop_mean'])} | "
            f"{_fmt(row['delta_logit_drop_mean'])} | "
            f"[{_fmt(row['delta_logit_ci95_low'])}, {_fmt(row['delta_logit_ci95_high'])}] | "
            f"{float(row['delta_logit_pvalue']):.4g} |"
        )

    interpretable = overall[
        (overall["delta_logit_drop_mean"] > 0)
        & (overall["target_logit_drop_mean"] > 0)
    ]
    if interpretable.empty:
        overall_take = "所有模型都没有同时满足 lesion drop 为正、delta 为正这两个条件，因此不能把整体 delta 解释为清晰的病灶区域敏感性。"
    else:
        names = ", ".join(interpretable["model"].tolist())
        overall_take = f"{names} 同时满足 lesion drop 为正、delta 为正，说明这些模型存在一定类别对齐的病灶区域敏感性；但仍需要结合 CI、p-value 和分疾病结果判断效应是否足够强。"

    best = interpretable.iloc[interpretable["delta_logit_drop_mean"].argmax()] if not interpretable.empty else overall.iloc[overall["delta_logit_drop_mean"].argmax()]
    mae_rows = overall[overall["model"].str.contains("MAE", regex=False)]
    mae_note = ""
    if not mae_rows.empty:
        mae_bad = mae_rows[mae_rows["target_logit_drop_mean"] < 0]
        if not mae_bad.empty:
            names = ", ".join(mae_bad["model"].tolist())
            mae_note = f"{names} 的 lesion drop 为负，表示遮挡后目标 logit 反而升高；其正 delta 只能说明 lesion 与 control 的相对差异，不能解释为更强 localization。"

    lines.extend(
        [
            "",
            overall_take,
            mae_note,
            f"在满足 lesion drop 为正的模型中，效应最大的模型是 {best['model']}，其 delta 为 {_fmt(best['delta_logit_drop_mean'])}。这个值应解释为 target-class logit 层面的额外下降，而不是 AUROC 或准确率的直接变化。",
            "",
            "![Fig. 10. Exp2b localized vs diffuse group effects.](results/paper_figures/fig10_exp2b_group_effects.png)",
            "",
            "| 模型 | 疾病组 | n | Delta logit drop | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in groups.iterrows():
        lines.append(
            f"| {row['model']} | {row['group']} | {int(row['n'])} | "
            f"{_fmt(row['delta_logit_drop_mean'])} | "
            f"[{_fmt(row['delta_logit_ci95_low'])}, {_fmt(row['delta_logit_ci95_high'])}] |"
        )

    lines.extend(
        [
            "",
            "分组结果是 Exp2b 的关键，因为胸片疾病并不都对应局部小病灶。`localized` 组更接近传统意义上的局部证据遮挡；`diffuse_or_global` 组包含心影增大、胸腔积液、间质性改变等更依赖全局结构或大范围纹理的标签。如果 localized 组强于 diffuse/global 组，说明模型确实在部分局部疾病上使用了 bbox evidence；如果两组都很弱，则说明 frozen SSL 表征的分类信息更可能来自全局上下文、共现关系或 bbox 外区域。",
            "",
            "![Fig. 11. Exp2b class heatmap.](results/paper_figures/fig11_exp2b_class_heatmap.png)",
            "",
            "![Fig. 12. Exp2b area response.](results/paper_figures/fig12_exp2b_area_response.png)",
            "",
            "面积分析用于排除一个重要混杂因素：遮挡越大，模型越容易变化，但这不一定是病灶特异性。如果 delta 随 bbox 面积单调增大，且 control 也同步增强，则说明实验主要反映 mask area sensitivity；如果在中小 bbox 的 localized 疾病上仍有正 delta，才更支持 lesion-specific evidence 的解释。",
            "",
            "![Fig. 13. Exp2b qualitative class-aligned cases.](results/paper_figures/fig13_exp2b_case_sheet.png)",
            "",
            "总体上，Exp2b 比旧 Exp2 更适合作为论文主实验，因为它把问题从“遮盖任意 bbox 是否影响整体分类”收紧为“遮盖某类疾病的标注区域是否影响该疾病本身”。因此无论结果是否强阳性，结论都会更清晰：强阳性支持 bbox-aware local evidence；弱阳性或阴性则说明高 clean AUROC 并不自动等价于局部病灶敏感性。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    overall, groups, classes = load_tables()
    section = build_markdown(overall, groups, classes)
    path = OUT / "exp2b_report_section.md"
    path.write_text(section, encoding="utf-8")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
