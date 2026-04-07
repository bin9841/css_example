from __future__ import annotations

import numpy as np
import pandas as pd

from src.candidate_features import generate_candidate_features
from src.label_builder import build_proxy_label_table_from_snapshot
from src.population_definition import run_population_definition
from src.scorecard_modeling import run_proxy_scorecard_case_analysis


def run_proxy_validation() -> dict[str, pd.DataFrame]:
    """Run provisional validation using the proxy snapshot label."""
    base_df = pd.read_csv(
        "retail_credit_synthetic_dataset_v3_with_exposure_flags.csv",
        encoding="utf-8-sig",
    )
    population_df, _, _ = run_population_definition()
    filtered_base = base_df.merge(
        population_df[["고객번호", "기준년월"]],
        on=["고객번호", "기준년월"],
        how="inner",
        validate="one_to_one",
    )
    feature_df = generate_candidate_features(filtered_base)
    label_df = build_proxy_label_table_from_snapshot(filtered_base)
    model_result = run_proxy_scorecard_case_analysis(feature_df=feature_df, label_df=label_df)

    dev_scored = _score_from_model_result(model_result["woe_transformed_dev"], model_result["coefficients"], "development")
    val_scored = _score_from_model_result(model_result["woe_transformed_validation"], model_result["coefficients"], "validation1")
    oot_scored = _score_from_model_result(model_result["woe_transformed_oot"], model_result["coefficients"], "validation2_oot")

    metric_summary = _build_metric_summary(dev_scored, val_scored, oot_scored)
    ranking_summary = _build_ranking_summary(dev_scored, val_scored, oot_scored)
    interpretability_review = _build_interpretability_review(model_result)
    risk_summary = _build_risk_summary(metric_summary, interpretability_review)

    return {
        "metric_summary": metric_summary,
        "ranking_summary": ranking_summary,
        "interpretability_review": interpretability_review,
        "risk_summary": risk_summary,
    }


def _score_from_model_result(woe_df: pd.DataFrame, coefficients_df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    factor = 40.0 / np.log(2.0)
    coef_map = coefficients_df.set_index("feature_name")["coefficient"].to_dict()
    feature_cols = [col for col in coefficients_df["feature_name"].tolist() if col != "const"]
    intercept = coef_map.get("const", 0.0)
    out = woe_df[["고객번호", "기준년월", "bad_flag"]].copy()
    linear_term = intercept
    for col in feature_cols:
        if col in woe_df.columns:
            linear_term = linear_term + coef_map[col] * woe_df[col]
    out["score"] = 500.0 - factor * linear_term
    out["period_name"] = period_name
    return out


def _auc(y_true: pd.Series, score: pd.Series) -> float:
    y = y_true.astype(int).to_numpy()
    s = score.to_numpy()
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n_pos = pos.sum()
    n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ks(y_true: pd.Series, score: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true.astype(int), "score": score}).sort_values("score", ascending=False)
    bad_total = df["y"].sum()
    good_total = len(df) - bad_total
    if bad_total == 0 or good_total == 0:
        return np.nan
    df["cum_bad"] = df["y"].cumsum() / bad_total
    df["cum_good"] = ((1 - df["y"]).cumsum()) / good_total
    return float((df["cum_bad"] - df["cum_good"]).abs().max())


def _make_score_bands(score_series: pd.Series, width: int = 40) -> list[float]:
    min_score = int(np.floor(score_series.min() / width) * width)
    max_score = int(np.ceil(score_series.max() / width) * width) + width
    return list(range(min_score, max_score + width, width))


def _band_distribution(sample_df: pd.DataFrame, band_edges: list[float]) -> pd.DataFrame:
    labels = [f">{int(edge)}" for edge in band_edges[:-1]]
    bands = pd.cut(
        sample_df["score"],
        bins=band_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    temp = sample_df.copy()
    temp["score_band"] = bands.astype(str)
    grouped = temp.groupby("score_band").agg(total_cnt=("bad_flag", "size"), bad_cnt=("bad_flag", "sum"))
    grouped["good_cnt"] = grouped["total_cnt"] - grouped["bad_cnt"]
    grouped["tot_pct"] = grouped["total_cnt"] / grouped["total_cnt"].sum()
    grouped["bad_rate"] = grouped["bad_cnt"] / grouped["total_cnt"].replace(0, np.nan)
    grouped["cum_bad"] = grouped["bad_cnt"].cumsum() / max(grouped["bad_cnt"].sum(), 1)
    grouped["cum_good"] = grouped["good_cnt"].cumsum() / max(grouped["good_cnt"].sum(), 1)
    grouped["ks"] = (grouped["cum_bad"] - grouped["cum_good"]).abs()
    return grouped.reset_index()


def _psi(base_pct: pd.Series, comp_pct: pd.Series) -> float:
    aligned = pd.concat([base_pct, comp_pct], axis=1).fillna(1e-6)
    aligned.columns = ["base", "comp"]
    aligned["base"] = aligned["base"].replace(0, 1e-6)
    aligned["comp"] = aligned["comp"].replace(0, 1e-6)
    return float(((aligned["base"] - aligned["comp"]) * np.log(aligned["base"] / aligned["comp"])).sum())


def _build_metric_summary(dev_df: pd.DataFrame, val_df: pd.DataFrame, oot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bands = _make_score_bands(dev_df["score"], width=40)
    base_dist = _band_distribution(dev_df, bands).set_index("score_band")["tot_pct"]
    for name, df in [("development", dev_df), ("validation1", val_df), ("validation2_oot", oot_df)]:
        auroc = _auc(df["bad_flag"], -df["score"])
        ks = _ks(df["bad_flag"], -df["score"])
        dist = _band_distribution(df, bands)
        psi = 0.0 if name == "development" else _psi(base_dist, dist.set_index("score_band")["tot_pct"])
        rows.append(
            {
                "period_name": name,
                "sample_cnt": len(df),
                "bad_cnt": int(df["bad_flag"].sum()),
                "good_cnt": int((1 - df["bad_flag"]).sum()),
                "bad_rate": float(df["bad_flag"].mean()),
                "psi": psi,
                "car_proxy": float(dist["bad_rate"].std()),
                "ks": ks,
                "roc": auroc,
                "cap_proxy": auroc,
                "sdr_proxy": float(dist["tot_pct"].std()),
                "cdr_proxy": float(dist["bad_rate"].fillna(0).std()),
                "ar": 2 * auroc - 1 if pd.notna(auroc) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_ranking_summary(dev_df: pd.DataFrame, val_df: pd.DataFrame, oot_df: pd.DataFrame) -> pd.DataFrame:
    bands = _make_score_bands(dev_df["score"], width=40)
    outputs = []
    for name, df in [("development", dev_df), ("validation1", val_df), ("validation2_oot", oot_df)]:
        band_df = _band_distribution(df, bands)
        band_df["period_name"] = name
        outputs.append(band_df)
    return pd.concat(outputs, ignore_index=True)


def _build_interpretability_review(model_result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    coef_df = model_result["coefficients"].copy()
    sign_df = model_result["sign_check"].copy()
    vif_df = model_result["vif"].copy()
    out = coef_df.merge(sign_df, on="feature_name", how="left").merge(vif_df, on="feature_name", how="left")
    out["regulatory_interpretability_flag"] = np.where(
        (out["feature_name"] == "const")
        | (
            out["sign_match_flag"].fillna("Y").eq("Y")
            & (out["p_value"].fillna(0) <= 0.05)
            & (out["vif"].fillna(0) < 5.0)
        ),
        "Y",
        "N",
    )
    return out


def _build_risk_summary(metric_summary: pd.DataFrame, interpretability_review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dev = metric_summary.loc[metric_summary["period_name"] == "development"].iloc[0]
    val1 = metric_summary.loc[metric_summary["period_name"] == "validation1"].iloc[0]
    val2 = metric_summary.loc[metric_summary["period_name"] == "validation2_oot"].iloc[0]
    rows.append(
        {
            "risk_topic": "stability",
            "observation": f"PSI vs development: validation1={val1['psi']:.4f}, validation2={val2['psi']:.4f}",
            "mitigation": "monitor drift and replace proxy label with forward 12M real label before final approval",
        }
    )
    rows.append(
        {
            "risk_topic": "discrimination",
            "observation": f"AR/KS development={dev['ar']:.4f}/{dev['ks']:.4f}, validation1={val1['ar']:.4f}/{val1['ks']:.4f}, validation2={val2['ar']:.4f}/{val2['ks']:.4f}",
            "mitigation": "re-run full validation with borrower-level forward performance data",
        }
    )
    flagged = interpretability_review.loc[interpretability_review["regulatory_interpretability_flag"] == "N"]
    rows.append(
        {
            "risk_topic": "interpretability",
            "observation": f"flagged coefficient rows={len(flagged)}",
            "mitigation": "review non-significant or unstable variables and refresh case selection if needed",
        }
    )
    rows.append(
        {
            "risk_topic": "label_validity",
            "observation": "current validation uses proxy snapshot label, not Basel-style forward 12M bad flag",
            "mitigation": "treat all current conclusions as provisional only",
        }
    )
    return pd.DataFrame(rows)
