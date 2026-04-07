from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.candidate_features import build_candidate_feature_list, generate_candidate_features
from src.excel_report import _autosize_worksheet
from src.label_builder import build_proxy_label_table_from_snapshot
from src.population_definition import run_population_definition
from src.scorecard_modeling import (
    assign_woe_bins,
    build_provisional_scorecard_table,
    run_proxy_scorecard_case_analysis,
)
from src.time_structure import build_split_summary, build_time_structure
from src.validation_framework import build_traffic_light_framework, build_validation_metric_table


DEFAULT_DETAILED_OUTPUT_PATH = Path("reports") / "retail_behavior_scorecard_detailed_process.xlsx"


def export_detailed_process_workbook(
    output_path: str | Path = DEFAULT_DETAILED_OUTPUT_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv("retail_credit_synthetic_dataset_v3_with_exposure_flags.csv", encoding="utf-8-sig")
    population_df, exclusion_df, waterfall_df = run_population_definition()
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

    valid_sheet = _build_validation_sheet(scored_dev, scored_val, scored_oot)
    period_summary = _build_period_comparison_summary(scored_dev, scored_val, scored_oot)
    variable_summary = _build_variable_summary(
        dev_bins=dev_bins,
        val_bins=val_bins,
        oot_bins=oot_bins,
        woe_maps=model_result["woe_maps"],
        feature_names=final_feature_names,
    )
    points_table = _build_points_table_detailed(
        dev_bins=dev_bins,
        val_bins=val_bins,
        oot_bins=oot_bins,
        woe_maps=model_result["woe_maps"],
        coefficients_df=model_result["coefficients"],
        feature_names=final_feature_names,
    )
    score_band_sheet = _build_score_band_sheet(scored_dev, scored_val, scored_oot, band_width=40)

    toc_df = pd.DataFrame(
        [
            {"order": 1, "process": "1. 개요", "sheet_name": "00_목차"},
            {"order": 2, "process": "2.1 모집단 정의", "sheet_name": "01_모집단정의"},
            {"order": 3, "process": "2.2 기간 구조 정의", "sheet_name": "02_기간구조"},
            {"order": 4, "process": "2.3 우불량 정의", "sheet_name": "03_우불량정의"},
            {"order": 5, "process": "2.4 모형 세분화", "sheet_name": "04_모형세분화"},
            {"order": 6, "process": "3.2 후보항목 추출", "sheet_name": "05_후보항목"},
            {"order": 7, "process": "3.3 유의성 분석", "sheet_name": "06_유의성분석"},
            {"order": 8, "process": "3.4 모형 적합 및 평점표 작성", "sheet_name": "07_모형적합"},
            {"order": 9, "process": "4.1 모형 검증", "sheet_name": "08_검증지표"},
            {"order": 10, "process": "기간 비교 요약", "sheet_name": "09_기간비교요약"},
            {"order": 11, "process": "사용 변수 요약", "sheet_name": "10_변수요약"},
            {"order": 12, "process": "평점표", "sheet_name": "11_평점표"},
            {"order": 13, "process": "40점 단위 평점분포", "sheet_name": "12_평점분포40"},
        ]
    )

    segmentation_df = pd.DataFrame(
        [
            {
                "model_type": "BS",
                "current_assumption": "단일 카드 BS 모형",
                "pdf_reference": "개인카드, 개인기업카드, 소호여신은 단일 모형",
                "note": "현재 데이터와 구현은 카드 BS 중심으로 정리됨",
            }
        ]
    )

    bad_def_df = label_df.head(0).copy()
    bad_def_df = pd.DataFrame(
        [
            {"item": "bad_definition_type", "value": "proxy_snapshot"},
            {"item": "warning", "value": "forward 12M borrower-level bad label not available"},
            {"item": "bad_rule", "value": "current delinquency or internal/external 60dpd history or bureau default proxy"},
            {"item": "indeterminate_rule", "value": "10-59 DPD history proxy"},
        ]
    )

    model_fit_df = pd.concat(
        [
            model_result["case_comparison"].assign(section="case_comparison"),
            model_result["coefficients"].assign(section="coefficients"),
            model_result["vif"].assign(section="vif"),
            model_result["sign_check"].assign(section="sign_check"),
        ],
        ignore_index=True,
        sort=False,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sheets = [
            ("00_목차", toc_df),
            ("01_모집단정의", population_df),
            ("01_1_Waterfall", waterfall_df),
            ("01_2_제외상세", exclusion_df),
            ("02_기간구조", build_time_structure()),
            ("02_1_Split", build_split_summary()),
            ("03_우불량정의", bad_def_df),
            ("04_모형세분화", segmentation_df),
            ("05_후보항목", build_candidate_feature_list()),
            ("05_1_후보변수", pd.DataFrame({"feature_name": final_feature_names})),
            ("06_유의성분석", variable_summary),
            ("07_모형적합", model_fit_df),
            ("08_검증지표", valid_sheet),
            ("08_1_교통신호", build_traffic_light_framework()),
            ("08_2_검증Metric", build_validation_metric_table()),
            ("09_기간비교요약", period_summary),
            ("10_변수요약", variable_summary),
            ("11_평점표", points_table),
            ("12_평점분포40", score_band_sheet),
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
    score_df["period_name"] = score_df["기준년월"].map(
        {202209: "validation1", 202309: "development", 202409: "validation2"}
    )
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


def _psi(base_pct: pd.Series, comp_pct: pd.Series) -> float:
    aligned = pd.concat([base_pct, comp_pct], axis=1).fillna(1e-6)
    aligned.columns = ["base", "comp"]
    aligned["base"] = aligned["base"].replace(0, 1e-6)
    aligned["comp"] = aligned["comp"].replace(0, 1e-6)
    return float(((aligned["base"] - aligned["comp"]) * np.log(aligned["base"] / aligned["comp"])).sum())


def _score_band_distribution(sample_df: pd.DataFrame, band_edges: list[float]) -> pd.DataFrame:
    band_labels = [f">{int(edge)}" for edge in band_edges[:-1]]
    bands = pd.cut(
        sample_df["score"],
        bins=band_edges,
        labels=band_labels,
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
    grouped["ks"] = (grouped["cum_bad_pct"] - grouped["cum_good_pct"]).abs()
    return grouped.reset_index()


def _build_validation_sheet(dev_df: pd.DataFrame, val_df: pd.DataFrame, oot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_dist = None
    for name, df in [("development", dev_df), ("validation1", val_df), ("validation2", oot_df)]:
        auroc = _auc(df["bad_flag"], -df["score"])
        ar = 2 * auroc - 1 if pd.notna(auroc) else np.nan
        ks = _ks(df["bad_flag"], -df["score"])
        rows.append(
            {
                "period": name,
                "rows": len(df),
                "bad_cnt": int(df["bad_flag"].sum()),
                "good_cnt": int((1 - df["bad_flag"]).sum()),
                "bad_rate": float(df["bad_flag"].mean()),
                "ks": ks,
                "auroc": auroc,
                "ar": ar,
            }
        )
    out = pd.DataFrame(rows)
    band_edges = _make_score_band_edges(dev_df["score"], step=40)
    base_dist = _score_band_distribution(dev_df, band_edges).set_index("score_band")["tot_pct"]
    for compare_name, compare_df in [("validation1", val_df), ("validation2", oot_df)]:
        compare_dist = _score_band_distribution(compare_df, band_edges).set_index("score_band")["tot_pct"]
        out.loc[out["period"] == compare_name, "psi_vs_dev"] = _psi(base_dist, compare_dist)
    out.loc[out["period"] == "development", "psi_vs_dev"] = 0.0
    return out


def _build_period_comparison_summary(dev_df: pd.DataFrame, val_df: pd.DataFrame, oot_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = _build_validation_sheet(dev_df, val_df, oot_df)
    return valid_df.copy()


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
        psi_val1 = _psi(dev_dist, val_dist)
        psi_val2 = _psi(dev_dist, oot_dist)
        car_val1 = _car_proxy(dev_bins, val_bins, feature_name)
        car_val2 = _car_proxy(dev_bins, oot_bins, feature_name)
        rows.append(
            {
                "feature_name": feature_name,
                "iv_dev": iv,
                "psi_validation1": psi_val1,
                "psi_validation2": psi_val2,
                "car_validation1_proxy": car_val1,
                "car_validation2_proxy": car_val2,
            }
        )
    return pd.DataFrame(rows).sort_values("iv_dev", ascending=False)


def _car_proxy(dev_bins: pd.DataFrame, compare_bins: pd.DataFrame, feature_name: str) -> float:
    bin_col = f"{feature_name}__bin"
    dev_bad_rate = dev_bins.groupby(bin_col)["bad_flag"].mean()
    comp_bad_rate = compare_bins.groupby(bin_col)["bad_flag"].mean()
    all_rates = pd.concat([dev_bad_rate, comp_bad_rate], axis=1).fillna(0.0)
    all_rates.columns = ["dev", "comp"]
    return float((all_rates["dev"] - all_rates["comp"]).abs().mean())


def _build_points_table_detailed(
    dev_bins: pd.DataFrame,
    val_bins: pd.DataFrame,
    oot_bins: pd.DataFrame,
    woe_maps: dict[str, dict[str, object]],
    coefficients_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    points_df = build_provisional_scorecard_table(woe_maps=woe_maps, coefficients_df=coefficients_df)
    rows = []
    for _, row in points_df.iterrows():
        feature_name = row["feature_name"]
        bin_name = row["bin"]
        dev_part = dev_bins.loc[dev_bins[f"{feature_name}__bin"] == bin_name]
        val_part = val_bins.loc[val_bins[f"{feature_name}__bin"] == bin_name]
        oot_part = oot_bins.loc[oot_bins[f"{feature_name}__bin"] == bin_name]
        dev_pct = len(dev_part) / max(len(dev_bins), 1)
        val_pct = len(val_part) / max(len(val_bins), 1)
        oot_pct = len(oot_part) / max(len(oot_bins), 1)
        rows.append(
            {
                "feature_name": feature_name,
                "bin": bin_name,
                "points": row["bin_points"],
                "woe": row["woe"],
                "iv_component": row["iv_component"],
                "dev_total_cnt": len(dev_part),
                "dev_total_pct": dev_pct,
                "dev_good_cnt": int((1 - dev_part["bad_flag"]).sum()),
                "dev_bad_cnt": int(dev_part["bad_flag"].sum()),
                "dev_bad_rate": float(dev_part["bad_flag"].mean()) if len(dev_part) else np.nan,
                "val1_total_cnt": len(val_part),
                "val1_total_pct": val_pct,
                "val2_total_cnt": len(oot_part),
                "val2_total_pct": oot_pct,
                "psi_val1": _psi(pd.Series([dev_pct]), pd.Series([val_pct])),
                "psi_val2": _psi(pd.Series([dev_pct]), pd.Series([oot_pct])),
            }
        )
    return pd.DataFrame(rows)


def _build_score_band_sheet(dev_df: pd.DataFrame, val_df: pd.DataFrame, oot_df: pd.DataFrame, band_width: int = 40) -> pd.DataFrame:
    edges = _make_score_band_edges(dev_df["score"], step=band_width)
    summaries = []
    base_dist = _score_band_distribution(dev_df, edges).set_index("score_band")["tot_pct"]
    for period_name, df in [("development", dev_df), ("validation1", val_df), ("validation2", oot_df)]:
        band_df = _score_band_distribution(df, edges)
        band_df["period"] = period_name
        auroc = _auc(df["bad_flag"], -df["score"])
        band_df["auroc"] = auroc
        band_df["ar"] = 2 * auroc - 1 if pd.notna(auroc) else np.nan
        band_df["overall_ks"] = _ks(df["bad_flag"], -df["score"])
        current_dist = band_df.set_index("score_band")["tot_pct"]
        band_df["psi_vs_dev"] = 0.0 if period_name == "development" else _psi(base_dist, current_dist)
        summaries.append(band_df)
    return pd.concat(summaries, ignore_index=True)


def _make_score_band_edges(score_series: pd.Series, step: int = 40) -> list[float]:
    min_score = int(np.floor(score_series.min() / step) * step)
    max_score = int(np.ceil(score_series.max() / step) * step) + step
    return list(range(min_score, max_score + step, step))
