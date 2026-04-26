from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("无法自动定位项目根目录，请在项目目录内运行该脚本。")


ROOT = find_project_root()
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"


def read_csmar_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, header=0, skiprows=[1, 2])


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(6)


def filter_annual_year_end(df: pd.DataFrame, date_col: str, start: int = 2010, end: int = 2025) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.loc[out[date_col].dt.strftime("%m-%d").eq("12-31")].copy()
    out["year"] = out[date_col].dt.year
    out = out.loc[out["year"].between(start, end)].copy()
    return out


def load_balance_sheet() -> pd.DataFrame:
    cols = [
        "Stkcd",
        "ShortName",
        "Accper",
        "Typrep",
        "A001100000",
        "A001212000",
        "A001000000",
        "A002100000",
        "A002000000",
    ]
    df = read_csmar_excel(RAW / "balance_sheet" / "balance_sheet.xlsx")[cols]
    # Keep 2009 year-end assets as a lag buffer so Growth in 2010 can be computed
    # with 2009 total assets, then trim the final panel back to 2010-2025 later.
    df = filter_annual_year_end(df, "Accper", start=2009, end=2025)
    df = df.loc[df["Typrep"].eq("A")].copy()
    df["stkcd"] = normalize_code(df["Stkcd"])
    df["short_name"] = df["ShortName"].astype(str)
    rename_map = {
        "A001100000": "current_assets",
        "A001212000": "fixed_assets_net",
        "A001000000": "total_assets",
        "A002100000": "current_liabilities",
        "A002000000": "total_liabilities",
    }
    df = df.rename(columns=rename_map)
    keep = ["stkcd", "short_name", "year", *rename_map.values()]
    df = df[keep].drop_duplicates(subset=["stkcd", "year"], keep="last")
    return df


def load_income_stmt() -> pd.DataFrame:
    cols = ["Stkcd", "Accper", "Typrep", "B002000000"]
    df = read_csmar_excel(RAW / "income_stmt" / "income_stmt.xlsx")[cols]
    df = filter_annual_year_end(df, "Accper")
    df = df.loc[df["Typrep"].eq("A")].copy()
    df["stkcd"] = normalize_code(df["Stkcd"])
    df = df.rename(columns={"B002000000": "net_profit"})
    df = df[["stkcd", "year", "net_profit"]].drop_duplicates(subset=["stkcd", "year"], keep="last")
    return df


def load_cashflow() -> pd.DataFrame:
    cols = [
        "Stkcd",
        "Accper",
        "Typrep",
        "D000103000",
        "D000119000",
        "D000120000",
        "D000104000",
        "D000105000",
    ]
    df = read_csmar_excel(RAW / "cashflow" / "cashflow.xlsx")[cols]
    df = filter_annual_year_end(df, "Accper")
    df = df.loc[df["Typrep"].eq("A")].copy()
    df["stkcd"] = normalize_code(df["Stkcd"])
    rename_map = {
        "D000103000": "dep_fixed_assets",
        "D000119000": "dep_investment_property",
        "D000120000": "dep_rou_assets",
        "D000104000": "amort_intangible",
        "D000105000": "amort_long_term_deferred",
    }
    df = df.rename(columns=rename_map)
    for col in rename_map.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["ndts_numerator"] = (
        df["dep_fixed_assets"] + df["amort_intangible"] + df["amort_long_term_deferred"]
    )
    keep = ["stkcd", "year", *rename_map.values(), "ndts_numerator"]
    df = df[keep].drop_duplicates(subset=["stkcd", "year"], keep="last")
    return df


def load_ownership() -> pd.DataFrame:
    cols = ["Symbol", "EndDate", "EquityNature"]
    df = read_csmar_excel(RAW / "ownership" / "ownership.xlsx")[cols]
    df = filter_annual_year_end(df, "EndDate")
    df["stkcd"] = normalize_code(df["Symbol"])
    df["equity_nature"] = df["EquityNature"].astype(str)
    df["soe"] = df["equity_nature"].str.contains("国企", na=False).astype("Int64")
    df = df[["stkcd", "year", "equity_nature", "soe"]].drop_duplicates(subset=["stkcd", "year"], keep="last")
    return df


def load_industry() -> pd.DataFrame:
    cols = ["Symbol", "EndDate", "IndustryCode"]
    df = read_csmar_excel(RAW / "industry" / "industry.xlsx")[cols]
    df = filter_annual_year_end(df, "EndDate")
    df["stkcd"] = normalize_code(df["Symbol"])
    df["industry_code"] = df["IndustryCode"].astype(str).str.strip()
    df["industry_group"] = np.where(
        df["industry_code"].str.startswith("C"),
        df["industry_code"].str[:3],
        df["industry_code"].str[:1],
    )
    df = df[["stkcd", "year", "industry_code", "industry_group"]].drop_duplicates(
        subset=["stkcd", "year"], keep="last"
    )
    return df


def load_st_flag() -> pd.DataFrame:
    cols = ["Symbol", "EndDate", "LISTINGSTATE"]
    df = read_csmar_excel(RAW / "st_flag" / "st_flag.xlsx")[cols]
    df = filter_annual_year_end(df, "EndDate")
    df["stkcd"] = normalize_code(df["Symbol"])
    df["listing_state"] = df["LISTINGSTATE"].astype(str).str.strip()
    st_pattern = r"ST|PT|暂停"
    df["is_stpt_year"] = df["listing_state"].str.contains(st_pattern, na=False).astype("Int64")
    df["ever_stpt"] = df.groupby("stkcd")["is_stpt_year"].transform("max").astype("Int64")
    df = df[["stkcd", "year", "listing_state", "is_stpt_year", "ever_stpt"]].drop_duplicates(
        subset=["stkcd", "year"], keep="last"
    )
    return df


def load_m2() -> pd.DataFrame:
    cols = ["Staper", "Efq0104"]
    df = read_csmar_excel(RAW / "m2" / "m2.xlsx")[cols]
    df["Staper"] = pd.to_datetime(df["Staper"], errors="coerce")
    df = df.loc[df["Staper"].dt.month.eq(12)].copy()
    df["year"] = df["Staper"].dt.year
    df = df.loc[df["year"].between(2010, 2025)].copy()
    df["m2_growth"] = pd.to_numeric(df["Efq0104"], errors="coerce")
    return df[["year", "m2_growth"]].drop_duplicates(subset=["year"], keep="last")


def add_constructed_variables(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["stkcd", "year"]).copy()

    numeric_cols = [
        "current_assets",
        "fixed_assets_net",
        "total_assets",
        "current_liabilities",
        "total_liabilities",
        "net_profit",
        "dep_fixed_assets",
        "dep_investment_property",
        "dep_rou_assets",
        "amort_intangible",
        "amort_long_term_deferred",
        "ndts_numerator",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["lev"] = df["total_liabilities"] / df["total_assets"]
    df["npr"] = df["net_profit"] / df["total_assets"]
    df["size"] = np.where(df["total_assets"] > 0, np.log(df["total_assets"]), np.nan)
    df["tang"] = df["fixed_assets_net"] / df["total_assets"]
    lag_assets = df.groupby("stkcd")["total_assets"].shift(1)
    df["growth"] = (df["total_assets"] - lag_assets) / lag_assets
    df["ndts"] = df["ndts_numerator"] / df["total_assets"]
    df["liq"] = df["current_assets"] / df["current_liabilities"]
    return df


def merge_small_manufacturing_groups(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    manuf = df["industry_group"].fillna("").str.startswith("C")
    counts = df.loc[manuf, "industry_group"].value_counts()
    small_groups = counts[counts < 30].index
    df["ind_code"] = df["industry_group"]
    df.loc[df["industry_group"].isin(small_groups), "ind_code"] = "C_other"
    return df


def build_panel() -> pd.DataFrame:
    balance = load_balance_sheet()
    income = load_income_stmt()
    cashflow = load_cashflow()
    ownership = load_ownership()
    industry = load_industry()
    st_flag = load_st_flag()
    m2 = load_m2()

    panel = balance.merge(income, on=["stkcd", "year"], how="outer")
    panel = panel.merge(cashflow, on=["stkcd", "year"], how="outer")
    panel = panel.merge(ownership, on=["stkcd", "year"], how="left")
    panel = panel.merge(industry, on=["stkcd", "year"], how="left")
    panel = panel.merge(st_flag, on=["stkcd", "year"], how="left")
    panel = panel.merge(m2, on="year", how="left")

    panel = add_constructed_variables(panel)
    panel = merge_small_manufacturing_groups(panel)
    panel = panel.loc[panel["year"].between(2010, 2025)].copy()

    keep_cols = [
        "stkcd",
        "short_name",
        "year",
        "lev",
        "npr",
        "size",
        "tang",
        "growth",
        "ndts",
        "liq",
        "soe",
        "equity_nature",
        "industry_code",
        "ind_code",
        "listing_state",
        "is_stpt_year",
        "ever_stpt",
        "m2_growth",
    ]
    panel = panel[keep_cols].sort_values(["stkcd", "year"]).reset_index(drop=True)
    return panel


def write_outputs(panel: pd.DataFrame) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    panel.to_csv(CLEAN / "panel_capital_structure_annual.csv", index=False, encoding="utf-8-sig")
    panel.to_parquet(CLEAN / "panel_capital_structure_annual.parquet", index=False)

    summary = [
        "# Panel Build Summary",
        "",
        f"- 行数: {len(panel):,}",
        f"- 公司数: {panel['stkcd'].nunique():,}",
        f"- 年份范围: {int(panel['year'].min())}-{int(panel['year'].max())}",
        f"- ever_stpt=1 的公司数: {panel.loc[panel['ever_stpt'].eq(1), 'stkcd'].nunique():,}",
    ]
    (CLEAN / "panel_capital_structure_annual_summary.md").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    panel_df = build_panel()
    write_outputs(panel_df)
