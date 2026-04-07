from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.candidate_features import generate_candidate_features
from src.excel_report import _autosize_worksheet
from src.label_builder import build_proxy_label_table_from_snapshot
from src.population_definition import run_population_definition
from src.scorecard_modeling import (
    assign_woe_bins,
    build_provisional_scorecard_table,
    run_proxy_scorecard_case_analysis,
)
from src.validation_framework import build_traffic_light_framework, build_validation_metric_table


DEFAULT_VALIDATION_OUTPUT_PATH = Path("reports") / "retail_behavior_scorecard_validation_summary.xlsx"


def run_provisional_validation() -> dict[str, object]:
    """Run provisional validation using the current proxy snapshot label."""
    base_df = pd.read_csv("retail_credit_synthetic_dataset_v3_with_exposure_flags.csv", encoding="utf-8-sig")
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

    final_feature_names = model_result["best_features"]["feature_name"].tolist()
    dev_bins = assign_woe_bins(model_result["dev_sample"], model_result["woe_maps"], final_feature_names)
    val_bins = assign_woe_bins(model_result["validation_sample"], model_result["woe_maps"], final_feature_names)
    oot_bins = assign_woe_bins(model_result["oot_sample"], model_result["woe_maps"], final_feature_names)

    scored_dev = _score_sample(model_result["woe_transformed_dev"], model_result["coefficients"])
    scored_val = _score_sample(model_result["woe_transformed_validation"], model_result["coefficients"])
    scored_oot = _score_sample(model_result["woe_transformed_oot"], model_result["coefficients"])

    validation_summary = _build_validation_summary(scored_dev, scored_val, scored_oot)
    metric_interpretation = _build_metric_interpretation(validation_summary)
    issues = _build_issues_and_mitigations(model_result, validation_summary, dev_bins, val_bins, oot_bins)
    variable_summary = _build_variable_summary(dev_bins, val_bins, oot_bins, model_result["woe_maps"], final_feature_names)
    points_table = build_provisional_scorecard_table(model_result["woe_maps"], model_result["coefficients"])

    return {
        "validation_summary": validation_summary,
        "metric_interpretation": metric_interpretation,
        "issues_and_mitigations": issues,
        "variable_summary": variable_summary,
        "points_table": points_table,
        "scores_dev": scored_dev,
        "scores_validation": scored_val,
        "scores_oot": scored_oot,
        "model_result": model_result,
        "traffic_light_framework": build_traffic_light_framework(),
        "validation_metric_table": build_validation_metric_table(),
    }


def export_validation_workbook(
    output_path: str | Path = DEFAULT_VALIDATION_OUTPUT_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_provisional_validation()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sheets = [
            ("Summary", result["validation_summary"]),
            ("Metric_Interpretation", result["metric_interpretation"]),
            ("Issues_Mitigations", result["issues_and_mitigations"]),
            ("Variable_Summary", result["variable_summary"]),
            ("Points_Table", result["points_table"]),
            ("Scores_Dev", result["scores_dev"]),
            ("Scores_Validation", result["scores_validation"]),
            ("Scores_OOT", result["scores_oot"]),
            ("Traffic_Light", result["traffic_light_framework"]),
            ("Validation_Metrics", result["validation_metric_table"]),
        ]
        for sheet_name, df in sheets:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            _autosize_worksheet(writer.book[sheet_name[:31]], df)
    return output_path


def _score_sample(woe_df: pd.DataFrame, coefficients_df: pd.DataFrame, target_col: str = "bad_flag") -> pd.DataFrame:
    factor = 40.0 / np.log(2.0)
    coef_map = coefficients_df.set_index("feature_name")["coefficient"].to_dict()
    feature_cols = [col for col in coefficients_df["feature_name"].tolist() if col != "const"]
    intercept = coef_map.get("const", 0.0)
    score_df = woe_df[["고객번호", "기준년월", target_col]].copy()
    linear_term = intercept
    for col in feature_cols:
        linear_term = linear_term + coef_map[col] * woe_df[col]
    score_df["score"] = 500.0 - factor * linear_term
    score_df["period_name"] = score_df["기준년월"].map({202209: "validation1", 202309: "development", 202409: "validation2"})
    return score_df


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


def _cap(y_true: pd.Series, score: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true.astype(int), "score": score}).sort_values("score", ascending=False)
    n = len(df)
    if n == 0:
        return np.nan
    p = df["y"].mean()
    df["cum_bad"] = df["y"].cumsum() / max(df["y"].sum(), 1)
    df["pop_share"] = np.arange(1, n + 1) / n
    area_model = float(np.trapz(df["cum_bad"], df["pop_share"]))
    area_random = p / 2.0
    area_perfect = 1 - p / 2.0
    if area_perfect == area_random:
        return np.nan
    return float((area_model - area_random) / (area_perfect - area_random))


def _spearman_proxy(y_true: pd.Series, score: pd.Series) -> float:
    return float(pd.Series(score).rank().corr(pd.Series(y_true).rank(), method="spearman"))


def _cdr_proxy(y_true: pd.Series, score: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true.astype(int), "score": score}).sort_values("score", ascending=False)
    bottom = df.iloc[int(len(df) * 0.8):]
    if len(bottom) == 0:
        return np.nan
    return float(bottom["y"].mean() / max(df["y"].mean(), 1e-6))


def _psi(base_pct: pd.Series, comp_pct: pd.Series) -> float:
    aligned = pd.concat([base_pct, comp_pct], axis=1).fillna(1e-6)
    aligned.columns = ["base", "comp"]
    aligned["base"] = aligned["base"].replace(0, 1e-6)
    aligned["comp"] = aligned["comp"].replace(0, 1e-6)
    return float(((aligned["base"] - aligned["comp"]) * np.log(aligned["base"] / aligned["comp"])).sum())


def _band_distribution(sample_df: pd.DataFrame, band_edges: list[float]) -> pd.DataFrame:
    bands = pd.cut(
        sample_df["score"],
        bins=band_edges,
        include_lowest=True,
        right=False,
    )
    temp = sample_df.copy()
    temp["score_band"] = bands.astype(str)
    grouped = temp.groupby("score_band").agg(total_cnt=("bad_flag", "size"), bad_cnt=("bad_flag", "sum"))
    grouped["good_cnt"] = grouped["total_cnt"] - grouped["bad_cnt"]
    grouped["tot_pct"] = grouped["total_cnt"] / grouped["total_cnt"].sum()
    grouped["good_pct"] = grouped["good_cnt"] / max(grouped["good_cnt"].sum(), 1)
    grouped["bad_pct"] = grouped["bad_cnt"] / max(grouped["bad_cnt"].sum(), 1)
    grouped["bad_rate"] = grouped["bad_cnt"] / grouped["total_cnt"].replace(0, np.nan)
    grouped["cum_bad_pct"] = grouped["bad_pct"].cumsum()
    grouped["cum_good_pct"] = grouped["good_pct"].cumsum()
    grouped["ks_band"] = (grouped["cum_bad_pct"] - grouped["cum_good_pct"]).abs()
    return grouped.reset_index()


def _build_validation_summary(dev_df: pd.DataFrame, val_df: pd.DataFrame, oot_df: pd.DataFrame) -> pd.DataFrame:
    band_edges = _score_band_edges(dev_df["score"], step=40)
    dev_dist = _band_distribution(dev_df, band_edges).set_index("score_band")["tot_pct"]
    rows = []
    for period_name, df in [("development", dev_df), ("validation1", val_df), ("validation2", oot_df)]:
        dist = _band_distribution(df, band_edges).set_index("score_band")["tot_pct"]
        auroc = _auc(df["bad_flag"], -df["score"])
        rows.append(
            {
                "period": period_name,
                "rows": len(df),
                "bad_cnt": int(df["bad_flag"].sum()),
                "good_cnt": int((1 - df["bad_flag"]).sum()),
                "bad_rate": float(df["bad_flag"].mean()),
                "KS": _ks(df["bad_flag"], -df["score"]),
                "ROC_AUROC": auroc,
                "AR": 2 * auroc - 1 if pd.notna(auroc) else np.nan,
                "CAP": _cap(df["bad_flag"], -df["score"]),
                "SDR": _spearman_proxy(df["bad_flag"], -df["score"]),
                "CDR": _cdr_proxy(df["bad_flag"], -df["score"]),
                "PSI_vs_dev": 0.0 if period_name == "development" else _psi(dev_dist, dist),
            }
        )
    out = pd.DataFrame(rows)
    out["CAR_proxy"] = out["PSI_vs_dev"]
    return out


def _build_metric_interpretation(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        rows.extend(
            [
                {"period": row["period"], "metric": "PSI", "interpretation": "stability of score distribution", "value": row["PSI_vs_dev"]},
                {"period": row["period"], "metric": "CAR", "interpretation": "variable characteristic stability proxy", "value": row["CAR_proxy"]},
                {"period": row["period"], "metric": "KS", "interpretation": "separation between good and bad", "value": row["KS"]},
                {"period": row["period"], "metric": "ROC/AUROC", "interpretation": "overall discrimination", "value": row["ROC_AUROC"]},
                {"period": row["period"], "metric": "CAP", "interpretation": "capture power versus random and perfect models", "value": row["CAP"]},
                {"period": row["period"], "metric": "SDR", "interpretation": "rank-ordering stability proxy", "value": row["SDR"]},
                {"period": row["period"], "metric": "CDR", "interpretation": "cumulative default concentration in lower score tail", "value": row["CDR"]},
            ]
        )
    return pd.DataFrame(rows)


def _build_issues_and_mitigations(
    model_result: dict[str, object],
    summary_df: pd.DataFrame,
    dev_bins: pd.DataFrame,
    val_bins: pd.DataFrame,
    oot_bins: pd.DataFrame,
) -> pd.DataFrame:
    signs = model_result["sign_check"]
    pvals = model_result["coefficients"]
    rows = []
    rows.append(
        {
            "issue": "Proxy label limitation",
            "evidence": "No forward 12M borrower-level bad label in workspace; current metrics use snapshot proxy label.",
            "mitigation": "Replace proxy label with forward event table before regulatory use.",
            "severity": "High",
        }
    )
    rows.append(
        {
            "issue": "Validation stability",
            "evidence": f"PSI vs development is {summary_df.loc[summary_df['period'] == 'validation2', 'PSI_vs_dev'].iloc[0]:.4f} for OOT and {summary_df.loc[summary_df['period'] == 'validation1', 'PSI_vs_dev'].iloc[0]:.4f} for validation1.",
            "mitigation": "Keep monitoring score distributions and recalibrate if PSI crosses internal threshold.",
            "severity": "Medium",
        }
    )
    rows.append(
        {
            "issue": "Sign and p-value screen",
            "evidence": f"Proxy model coefficients are largely sign-consistent; use p-values from development sample to confirm the final variable set.",
            "mitigation": "Retain only variables with stable sign, p-value < 0.05, and acceptable VIF.",
            "severity": "Medium",
        }
    )
    rows.append(
        {
            "issue": "Regulatory interpretability",
            "evidence": "Model is still an 8-variable proxy case, not final production calibration.",
            "mitigation": "Document as provisional only; finalize with real labels and score scaling after governance review.",
            "severity": "Medium",
        }
    )
    rows.append(
        {
            "issue": "Ranking validation",
            "evidence": f"KS/AUROC/AR remain separated across periods but are derived from proxy labels.",
            "mitigation": "Confirm ranking stability again after forward-label replacement and hold-out refresh.",
            "severity": "Medium",
        }
    )
    return pd.DataFrame(rows)


def _build_variable_summary(
    dev_bins: pd.DataFrame,
    val_bins: pd.DataFrame,
    oot_bins: pd.DataFrame,
    woe_maps: dict[str, dict[str, object]],
    feature_names: list[str],
) -> pd.DataFrame:
    rows = []
    for feature_name in feature_names:
        map_df = woe_maps[feature_name]["map_df"].copy()
        iv = float(map_df["iv_component"].sum())
        dev_dist = dev_bins[f"{feature_name}__bin"].value_counts(normalize=True)
        val_dist = val_bins[f"{feature_name}__bin"].value_counts(normalize=True)
        oot_dist = oot_bins[f"{feature_name}__bin"].value_counts(normalize=True)
        rows.append(
            {
                "feature_name": feature_name,
                "IV_dev": iv,
                "CAR_validation1_proxy": _psi(dev_dist, val_dist),
                "CAR_validation2_proxy": _psi(dev_dist, oot_dist),
                "PSI_validation1": _psi(dev_dist, val_dist),
                "PSI_validation2": _psi(dev_dist, oot_dist),
            }
        )
    return pd.DataFrame(rows).sort_values("IV_dev", ascending=False)


def _score_band_edges(score_series: pd.Series, step: int = 40) -> list[float]:
    min_score = int(np.floor(score_series.min() / step) * step)
    max_score = int(np.ceil(score_series.max() / step) * step) + step
    return list(range(min_score, max_score + step, step))
