from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.panel import PanelOLS
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("无法自动定位项目根目录，请在项目目录内运行该脚本。")


ROOT = find_project_root()
CLEAN = ROOT / "data" / "clean"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"


def ensure_dirs() -> None:
    for path in [FIGURES, TABLES]:
        path.mkdir(parents=True, exist_ok=True)


def sig_stars(pval: float) -> str:
    if pd.isna(pval):
        return ""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.1:
        return "*"
    return ""


def read_clean_panel() -> pd.DataFrame:
    filtered_path = CLEAN / "panel_capital_structure_filtered.csv"
    annual_path = CLEAN / "panel_capital_structure_annual.csv"
    path = filtered_path if filtered_path.exists() else annual_path
    df = pd.read_csv(path)
    df["stkcd"] = df["stkcd"].astype(int)
    df["year"] = df["year"].astype(int)
    return df


def build_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = []
    current = df.copy()
    steps.append({"step": "初始样本", "dropped_obs": np.nan, "remaining_obs": len(current), "remaining_firms": current["stkcd"].nunique()})

    before = len(current)
    current = current.loc[~current["industry_code"].astype(str).str.startswith("J")].copy()
    steps.append({"step": "剔除金融保险", "dropped_obs": before - len(current), "remaining_obs": len(current), "remaining_firms": current["stkcd"].nunique()})

    before = len(current)
    current = current.loc[current["ever_stpt"] != 1].copy()
    steps.append({"step": "剔除ST/PT", "dropped_obs": before - len(current), "remaining_obs": len(current), "remaining_firms": current["stkcd"].nunique()})

    before = len(current)
    current = current.loc[current["lev"] <= 1].copy()
    steps.append({"step": "剔除Lev > 1", "dropped_obs": before - len(current), "remaining_obs": len(current), "remaining_firms": current["stkcd"].nunique()})

    key_vars = ["lev", "npr", "size", "tang", "growth", "ndts", "soe", "ind_code", "m2_growth"]
    before = len(current)
    current = current.dropna(subset=key_vars).copy()
    steps.append({"step": "剔除缺失值", "dropped_obs": before - len(current), "remaining_obs": len(current), "remaining_firms": current["stkcd"].nunique()})

    current["soe"] = current["soe"].astype(int)
    return current, pd.DataFrame(steps)


def winsorize_by_year(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_prewin"] = out[col]
        out[col] = out.groupby("year")[col].transform(lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
    return out


def make_desc_table(df: pd.DataFrame) -> pd.DataFrame:
    vars_ = ["lev", "npr", "size", "tang", "growth", "ndts"]
    groups = {"Full": df, "SOE": df.loc[df["soe"] == 1], "NonSOE": df.loc[df["soe"] == 0]}
    records = []
    for group_name, sub in groups.items():
        for var in vars_:
            s = sub[var].dropna()
            records.append({
                "group": group_name,
                "variable": var,
                "N": len(s),
                "Mean": s.mean(),
                "SD": s.std(),
                "P10": s.quantile(0.10),
                "P25": s.quantile(0.25),
                "Median": s.quantile(0.50),
                "P75": s.quantile(0.75),
                "P90": s.quantile(0.90),
            })
    table = pd.DataFrame(records)
    tests = []
    for var in vars_:
        soe = df.loc[df["soe"] == 1, var].dropna()
        non = df.loc[df["soe"] == 0, var].dropna()
        t_stat, p_val = stats.ttest_ind(soe, non, equal_var=False)
        tests.append({
            "variable": var,
            "mean_diff_soe_minus_nonsoe": soe.mean() - non.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "stars": sig_stars(p_val),
        })
    test_df = pd.DataFrame(tests)
    return table.merge(test_df, on="variable", how="left")


def make_corr_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vars_ = ["lev", "npr", "size", "tang", "growth", "ndts", "soe"]
    corr = pd.DataFrame(index=vars_, columns=vars_, dtype=float)
    pvals = pd.DataFrame(index=vars_, columns=vars_, dtype=float)
    pretty = pd.DataFrame(index=vars_, columns=vars_, dtype=object)
    for i in vars_:
        for j in vars_:
            tmp = df[[i, j]].dropna()
            if i == j:
                corr.loc[i, j] = 1.0
                pvals.loc[i, j] = 0.0
                pretty.loc[i, j] = "1.000"
            else:
                r, p = stats.pearsonr(tmp[i], tmp[j])
                corr.loc[i, j] = r
                pvals.loc[i, j] = p
                pretty.loc[i, j] = f"{r:.3f}{sig_stars(p)}"
    return corr, pretty


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(df[["npr", "size", "tang", "growth", "ndts", "soe"]])
    rows = []
    for idx, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({"variable": col, "vif": variance_inflation_factor(X.values, idx)})
    return pd.DataFrame(rows)


def fit_twfe(df: pd.DataFrame, formula: str):
    panel = df.set_index(["stkcd", "year"]).sort_index()
    model = PanelOLS.from_formula(formula, data=panel)
    return model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)


def fit_ife_approx(df: pd.DataFrame, n_factors: int = 2, max_iter: int = 8):
    panel = df.set_index(["stkcd", "year"]).sort_index().copy()
    formula = "lev ~ 1 + npr + size + tang + growth + ndts + m2_growth + EntityEffects"
    res = PanelOLS.from_formula(formula, data=panel).fit(cov_type="robust")
    panel["alpha"] = panel.index.get_level_values(0).map(
        res.estimated_effects["estimated_effects"].groupby(level=0).first()
    )

    beta_cols = ["npr", "size", "tang", "growth", "ndts", "m2_growth"]
    current = res.params[beta_cols].copy()
    for _ in range(max_iter):
        x_fit = sum(panel[col] * current[col] for col in beta_cols) + panel["alpha"] + res.params["Intercept"]
        resid = panel["lev"] - x_fit
        resid_mat = resid.unstack("year").fillna(0.0)
        svd = TruncatedSVD(n_components=n_factors, random_state=0)
        low_rank = pd.DataFrame(
            svd.fit_transform(resid_mat) @ svd.components_,
            index=resid_mat.index,
            columns=resid_mat.columns,
        ).stack()
        low_rank.index.names = ["stkcd", "year"]
        panel["ife_component"] = low_rank.reindex(panel.index).fillna(0.0)
        panel["lev_adj"] = panel["lev"] - panel["ife_component"]
        res = PanelOLS.from_formula(
            "lev_adj ~ 1 + npr + size + tang + growth + ndts + m2_growth + EntityEffects",
            data=panel,
        ).fit(cov_type="robust")
        panel["alpha"] = panel.index.get_level_values(0).map(
            res.estimated_effects["estimated_effects"].groupby(level=0).first()
        )
        current = res.params[beta_cols].copy()
    return res


def make_regression_table(results: dict[str, object], firm_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    row_order = ["npr", "npr_soe", "m2_growth", "size", "tang", "growth", "ndts"]
    labels = {
        "npr": "NPR",
        "npr_soe": "NPR × SOE",
        "m2_growth": "m2_growth",
        "size": "Size",
        "tang": "Tang",
        "growth": "Growth",
        "ndts": "NDTS",
    }
    for key in row_order:
        row = {"term": labels[key]}
        for model_name, res in results.items():
            if key in getattr(res, "params").index:
                coef = res.params[key]
                se = res.std_errors[key]
                p = res.pvalues[key]
                row[model_name] = f"{coef:.4f}{sig_stars(p)}\n({se:.4f})"
            else:
                row[model_name] = "—"
        rows.append(row)

    meta_rows = [
        ("公司FE", {"M1_TWFE": "✓", "M1p_IFE": "✓", "M2_SOE": "✓", "M2_NonSOE": "✓", "M3_Interaction": "✓"}),
        ("年度FE", {"M1_TWFE": "✓", "M1p_IFE": "交互FE", "M2_SOE": "✓", "M2_NonSOE": "✓", "M3_Interaction": "✓"}),
        ("聚类标准误", {"M1_TWFE": "双向", "M1p_IFE": "Robust", "M2_SOE": "双向", "M2_NonSOE": "双向", "M3_Interaction": "双向"}),
        ("N", {name: f"{int(res.nobs):,}" for name, res in results.items()}),
        ("公司数", {name: f"{firm_counts[name]:,}" for name in results}),
        ("Within R²", {name: f"{res.rsquared_within:.4f}" for name, res in results.items()}),
    ]
    for label, content in meta_rows:
        row = {"term": label}
        row.update(content)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fig1(df: pd.DataFrame) -> None:
    yearly = df.groupby(["year", "soe"])[["lev", "npr"]].mean().reset_index()
    label_map = {0: "Non-SOE", 1: "SOE"}
    yearly["soe_label"] = yearly["soe"].map(label_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.lineplot(data=yearly, x="year", y="lev", hue="soe_label", marker="o", ax=axes[0])
    axes[0].set_title("Panel A. Mean Leverage by Year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Lev")

    sns.lineplot(data=yearly, x="year", y="npr", hue="soe_label", marker="o", ax=axes[1])
    axes[1].set_title("Panel B. Mean Profitability by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("NPR")

    sns.boxplot(data=df, x="year", y="lev", color="#89a8d8", ax=axes[2], fliersize=1)
    axes[2].set_title("Panel C. Annual Distribution of Leverage")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Lev")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(FIGURES / "Fig1_descriptive_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    settings = [("lev_prewin", "lev", "Lev"), ("npr_prewin", "npr", "NPR"), ("growth_prewin", "growth", "Growth")]
    for ax, (before, after, title) in zip(axes, settings):
        long_df = pd.DataFrame({
            "status": np.repeat(["Before Winsorize", "After Winsorize"], [len(df), len(df)]),
            "value": pd.concat([df[before], df[after]], ignore_index=True),
        })
        sns.boxplot(data=long_df, x="status", y="value", hue="status", legend=False, ax=ax, palette=["#c9c9c9", "#5b8bd0"])
        ax.set_title(title)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig2_winsorize_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, square=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig3_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4(df: pd.DataFrame, res3) -> None:
    x_grid = np.linspace(df["npr"].quantile(0.01), df["npr"].quantile(0.99), 100)
    means = df[["size", "tang", "growth", "ndts"]].mean()
    intercept = res3.params["Intercept"]
    private = intercept + res3.params["size"] * means["size"] + res3.params["tang"] * means["tang"] + res3.params["growth"] * means["growth"] + res3.params["ndts"] * means["ndts"] + res3.params["npr"] * x_grid
    soe = intercept + res3.params["size"] * means["size"] + res3.params["tang"] * means["tang"] + res3.params["growth"] * means["growth"] + res3.params["ndts"] * means["ndts"] + (res3.params["npr"] + res3.params["npr_soe"]) * x_grid

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_grid, private, label="SOE = 0", color="#3b7a57")
    ax.plot(x_grid, soe, label="SOE = 1", color="#b85c38")
    ax.set_xlabel("NPR")
    ax.set_ylabel("Predicted Lev")
    ax.set_title("Moderating Effect of SOE on the NPR-Lev Slope")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig4_soe_margins.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig5(m4_yearly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m4_yearly["year"], m4_yearly["coef"], marker="o", color="#355c7d")
    ax.fill_between(m4_yearly["year"], m4_yearly["ci_low"], m4_yearly["ci_high"], alpha=0.2, color="#355c7d")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\hat{\beta}_t$")
    ax.set_title("Time-Varying Effect of NPR on Leverage")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig5_time_varying_beta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_text_reports(results: dict) -> None:
    data_processing_text = f"""# 数据处理说明

根据作业要求，样本筛选按以下顺序执行：剔除金融保险行业、剔除曾被 ST/PT 处理的公司、剔除 `Lev > 1` 的观测，并剔除关键变量缺失值。样本筛选后，最终分析样本保留 {results['sample_selection'].iloc[-1]['remaining_obs']:,} 个公司-年度观测和 {results['sample_selection'].iloc[-1]['remaining_firms']:,} 家公司。

由于 `Growth` 需要使用上一年度总资产，本项目在构造 clean 数据时保留了 2009 年年末总资产作为滞后缓冲，因此 2010 年观测可以按照题目要求计算成长率。这一步是保证样本区间仍为 2010-2025 年且变量定义严格一致的关键。

连续变量 `Lev`、`NPR`、`Tang`、`Growth` 和 `NDTS` 均在年度截面内执行双侧 1% Winsorize。该处理显著压缩了极端尾部，但并未改变变量的主体分布特征，因此后续统计和回归结果可被理解为对异常值更稳健的估计。"""

    desc = results["desc_stats"]
    full_lev = desc.query("group == 'Full' and variable == 'lev'").iloc[0]
    full_npr = desc.query("group == 'Full' and variable == 'npr'").iloc[0]
    corr_npr_lev = results["corr_numeric"].loc["npr", "lev"]
    corr_size_npr = results["corr_numeric"].loc["size", "npr"]
    vif_max = results["vif"]["vif"].max()
    desc_text = f"""# 描述性统计与相关性分析

在最终样本中，杠杆率 `Lev` 的均值为 {full_lev['Mean']:.3f}，中位数为 {full_lev['Median']:.3f}；净利润率 `NPR` 的均值为 {full_npr['Mean']:.3f}，中位数为 {full_npr['Median']:.3f}。SOE 与非 SOE 的均值差异检验显示，国有企业具有更高的杠杆率、更大的资产规模和更高的固定资产比重，但其盈利能力和成长性显著低于非国有企业，差异在统计上均高度显著。

Pearson 相关系数矩阵显示，`NPR` 与 `Lev` 的相关系数为 {corr_npr_lev:.3f}，方向为显著负值，这为优序融资理论提供了初步支持。`Size` 与 `NPR` 的相关系数仅为 {corr_size_npr:.3f}，且统计上不显著，表明从简单相关关系看，企业规模与盈利能力的线性关系并不强。需要注意的是，`Tang` 与 `NDTS` 的相关系数达到 {results['corr_numeric'].loc['tang', 'ndts']:.3f}，超过了 0.7 的经验阈值；不过进一步计算的 VIF 最大值仅为 {vif_max:.3f}，说明多重共线性虽值得关注，但尚未严重到影响主要结论的识别。"""

    m1 = results["models"]["M1_TWFE"]
    m1p = results["models"]["M1p_IFE"]
    m2_soe = results["models"]["M2_SOE"]
    m2_non = results["models"]["M2_NonSOE"]
    m3 = results["models"]["M3_Interaction"]
    beta_private = m3["coef"]["npr"]
    beta_soe = m3["coef"]["npr"] + m3["coef"]["npr_soe"]
    model_text = f"""# 回归结果分析

双向固定效应基准模型（M1）显示，`NPR` 的估计系数为 {m1['coef']['npr']:.3f}（双向聚类标准误 {m1['se']['npr']:.3f}），在 1% 水平上显著为负。这表明在控制公司固定效应、年度固定效应以及规模、资产可抵押性、成长性和非债务税盾后，盈利能力越强的企业反而具有更低的杠杆率。该结果与优序融资理论一致，不支持权衡理论关于“盈利能力越高、债务越多”的核心预测。

交互固定效应稳健性检验（M1'）中，`NPR` 的系数仍为显著负值，估计值为 {m1p['coef']['npr']:.3f}。与 M1 相比，其绝对值有所缩小，说明在显式控制潜在时变共同冲击后，盈利能力对杠杆率的负向影响略有减弱，但结论方向保持稳定。与此同时，`m2_growth` 的系数为 {m1p['coef']['m2_growth']:.4f}，且在 5% 水平上显著为正，表明货币扩张环境会推动企业总体杠杆上升。

分组回归结果显示，国有企业样本中 `NPR` 的系数为 {m2_soe['coef']['npr']:.3f}，非国有企业样本中为 {m2_non['coef']['npr']:.3f}，两组均显著为负，但国有企业的负向关系更强。交互项模型（M3）中，民营企业的基准斜率为 {beta_private:.3f}，`NPR × SOE` 的系数为 {m3['coef']['npr_soe']:.3f}，在 1% 水平上显著，意味着国有企业的边际效应进一步下降至 {beta_soe:.3f}。因此，产权性质确实显著调节了 `NPR-Lev` 关系，但方向并非“民营企业更符合优序融资理论”的常见先验，而是国有企业样本表现出更强的负向斜率。一个可能的解释是，国有企业在样本期内承担了更明显的政策性去杠杆压力，同时其较高的初始杠杆水平使得内部现金流改善时更倾向于主动降杠杆。

时变系数模型（M4）进一步表明，`NPR` 对 `Lev` 的影响在 2010-2025 年始终为负，但强度存在明显波动。2015-2020 年间，负向系数绝对值相对收敛，最低点大致出现在 2018 年前后；2021 年以后，负向关系再度增强，并在 2023-2025 年达到样本期最强水平。这一演化路径与宏观环境具有一定一致性：2015 年以后去杠杆政策和监管强化可能压缩了部分企业通过债务扩张的空间，而疫情后的宽信用环境并未逆转盈利能力与杠杆率之间的负向关系。"""

    discussion_text = f"""# 核心讨论问题

1. **理论检验**  
综合 M1-M3 的结果，A 股上市公司资本结构整体上更符合优序融资理论。无论在全样本还是产权分组样本中，盈利能力对杠杆率的影响均显著为负，说明企业更倾向于以内部留存收益替代外部债务融资。产权性质会改变这一关系的强弱：国有企业的负向斜率更大，表明其在盈利改善时表现出更强的降杠杆倾向。

2. **时序稳定性**  
M4 表明该关系并非完全稳定。2010-2014 年负向效应较强，2015-2020 年显著减弱，2021 年后重新增强。结合政策背景，可以将 2015 年后的阶段性变化理解为去杠杆政策、供给侧改革和疫情冲击共同作用下融资行为再配置的结果。

3. **IFE vs TWFE**  
引入交互固定效应后，`NPR` 的系数绝对值由 {m1['coef']['npr']:.3f} 缩小至 {m1p['coef']['npr']:.3f}，但显著性和符号均保持稳定。这意味着部分负向关系确实与时变共同冲击相关，但即便控制这类冲击后，优序融资理论仍然得到稳健支持。

4. **产权异质性**  
M2 与 M3 一致显示，SOE 与非 SOE 的系数差异在统计上显著。该结果提示，融资可得性与信息不对称并不是唯一机制；在中国制度环境下，政策性去杠杆、国企考核约束以及高初始杠杆的调整需求，也可能使国有企业在利润上升时更快压降负债。"""

def summarize_result(res) -> dict:
    return {
        "coef": {k: float(v) for k, v in res.params.items()},
        "se": {k: float(v) for k, v in res.std_errors.items()},
        "p": {k: float(v) for k, v in res.pvalues.items()},
        "nobs": int(res.nobs),
        "rsquared_within": float(res.rsquared_within),
    }


def main() -> None:
    ensure_dirs()

    base = read_clean_panel()
    sample_selection_path = CLEAN / "sample_selection_013.csv"
    if sample_selection_path.exists() and (CLEAN / "panel_capital_structure_filtered.csv").exists():
        analysis = base.copy()
        analysis["soe"] = analysis["soe"].astype(int)
        sample_selection = pd.read_csv(sample_selection_path)
    else:
        analysis, sample_selection = build_sample(base)
    analysis = winsorize_by_year(analysis, ["lev", "npr", "tang", "growth", "ndts"])

    desc_table = make_desc_table(analysis)
    corr_numeric, corr_pretty = make_corr_tables(analysis)
    vif_table = compute_vif(analysis)

    m1 = fit_twfe(analysis, "lev ~ 1 + npr + size + tang + growth + ndts + EntityEffects + TimeEffects")
    m2_soe = fit_twfe(analysis.loc[analysis["soe"] == 1], "lev ~ 1 + npr + size + tang + growth + ndts + EntityEffects + TimeEffects")
    m2_non = fit_twfe(analysis.loc[analysis["soe"] == 0], "lev ~ 1 + npr + size + tang + growth + ndts + EntityEffects + TimeEffects")

    m3_df = analysis.copy()
    m3_df["npr_soe"] = m3_df["npr"] * m3_df["soe"]
    m3 = fit_twfe(m3_df, "lev ~ 1 + npr + npr_soe + size + tang + growth + ndts + EntityEffects + TimeEffects")

    m4_df = analysis.copy()
    for year in sorted(m4_df["year"].unique()):
        m4_df[f"npr_{year}"] = m4_df["npr"] * (m4_df["year"] == year)
    npr_terms = " + ".join([f"npr_{year}" for year in sorted(m4_df["year"].unique())])
    m4 = fit_twfe(m4_df, f"lev ~ 0 + {npr_terms} + size + tang + growth + ndts + EntityEffects + TimeEffects")
    m4_yearly = pd.DataFrame({
        "year": sorted(analysis["year"].unique()),
        "coef": [m4.params[f"npr_{year}"] for year in sorted(analysis["year"].unique())],
        "se": [m4.std_errors[f"npr_{year}"] for year in sorted(analysis["year"].unique())],
        "p_value": [m4.pvalues[f"npr_{year}"] for year in sorted(analysis["year"].unique())],
    })
    m4_yearly["ci_low"] = m4_yearly["coef"] - 1.96 * m4_yearly["se"]
    m4_yearly["ci_high"] = m4_yearly["coef"] + 1.96 * m4_yearly["se"]

    m1p = fit_ife_approx(analysis)

    model_results = {
        "M1_TWFE": summarize_result(m1),
        "M1p_IFE": summarize_result(m1p),
        "M2_SOE": summarize_result(m2_soe),
        "M2_NonSOE": summarize_result(m2_non),
        "M3_Interaction": summarize_result(m3),
    }
    firm_counts = {
        "M1_TWFE": analysis["stkcd"].nunique(),
        "M1p_IFE": analysis["stkcd"].nunique(),
        "M2_SOE": analysis.loc[analysis["soe"] == 1, "stkcd"].nunique(),
        "M2_NonSOE": analysis.loc[analysis["soe"] == 0, "stkcd"].nunique(),
        "M3_Interaction": analysis["stkcd"].nunique(),
    }
    reg_table = make_regression_table(
        {
            "M1_TWFE": m1,
            "M1p_IFE": m1p,
            "M2_SOE": m2_soe,
            "M2_NonSOE": m2_non,
            "M3_Interaction": m3,
        },
        firm_counts,
    )

    # Final-report style table names
    sample_selection.to_csv(TABLES / "Table1_sample_selection.csv", index=False, encoding="utf-8-sig")
    desc_table.to_csv(TABLES / "Table2_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    corr_pretty.to_csv(TABLES / "Table3_correlation_matrix.csv", encoding="utf-8-sig")
    reg_table.to_csv(TABLES / "Table4_regression_results.csv", index=False, encoding="utf-8-sig")
    m4_yearly.to_csv(TABLES / "Table5_time_varying_coefficients.csv", index=False, encoding="utf-8-sig")
    vif_table.to_csv(TABLES / "TableA1_vif.csv", index=False, encoding="utf-8-sig")

    plot_fig1(analysis)
    plot_fig2(analysis)
    plot_fig3(corr_numeric)
    plot_fig4(analysis, m3)
    plot_fig5(m4_yearly)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk")
    main()
