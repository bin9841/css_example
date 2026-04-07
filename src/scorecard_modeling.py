from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    sm = None


@dataclass(frozen=True)
class ModelingStep:
    step_no: int
    step_name: str
    objective: str
    output: str


@dataclass(frozen=True)
class ModelCase:
    case_id: str
    case_name: str
    variable_count_target: int
    composition_rule: str
    use_case: str


MODELING_WORKFLOW: tuple[ModelingStep, ...] = (
    ModelingStep(1, "coarse_classing", "freeze scorecard-ready bins on development sample", "binning_map"),
    ModelingStep(2, "woe_transformation", "convert coarse bins into monotonic WoE predictors", "woe_dataset"),
    ModelingStep(3, "case_building", "assemble 10-12 variable candidate cases by domain balance", "model_cases"),
    ModelingStep(4, "logistic_regression", "fit interpretable scorecard logistic models", "fitted_models"),
    ModelingStep(5, "diagnostic_review", "check p-value, VIF, sign consistency, and business interpretability", "diagnostic_summary"),
    ModelingStep(6, "case_selection", "choose final scorecard case balancing performance and governance", "final_variable_set"),
)


MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase("C1", "Balanced Core", 10, "1-2 variables per information domain with strongest stable IV", "baseline scorecard"),
    ModelCase("C2", "Behavior Heavy", 11, "weight internal delinquency, card usage, and salary stability more heavily", "behavior-dominant scorecard"),
    ModelCase("C3", "Conservative Stable", 12, "prefer lower-missing and more stable variables even at some IV sacrifice", "governance-focused scorecard"),
)


FINAL_VARIABLE_SET: tuple[dict[str, str], ...] = (
    {"feature_name": "int_dq_days_max_12m", "domain": "internal_delinquency", "expected_sign": "+"},
    {"feature_name": "int_dq_recency_3m_flag", "domain": "internal_delinquency", "expected_sign": "+"},
    {"feature_name": "deposit_liquidity_share_3m", "domain": "deposit_performance", "expected_sign": "-"},
    {"feature_name": "salary_transfer_freq_12m", "domain": "salary_transfer", "expected_sign": "-"},
    {"feature_name": "salary_transfer_avg_amt_6m", "domain": "salary_transfer", "expected_sign": "-"},
    {"feature_name": "cb_purchase_util_avg_3m", "domain": "cb_card_usage", "expected_sign": "+"},
    {"feature_name": "cb_cash_service_freq_6m", "domain": "cb_card_usage", "expected_sign": "+"},
    {"feature_name": "ext_cb_default_flag", "domain": "external_delinquency", "expected_sign": "+"},
    {"feature_name": "loan_exposure_nonbank_share", "domain": "loan_exposure", "expected_sign": "+"},
    {"feature_name": "loan_new_lender_freq_12m", "domain": "loan_exposure", "expected_sign": "+"},
    {"feature_name": "acct_tenure_months", "domain": "account_opening_history", "expected_sign": "-"},
)


COEFFICIENT_INTERPRETATION: tuple[dict[str, str], ...] = (
    {"feature_name": "int_dq_days_max_12m", "interpretation": "higher past internal delinquency severity should increase odds of future bad"},
    {"feature_name": "deposit_liquidity_share_3m", "interpretation": "higher liquid deposit support should reduce future bad risk"},
    {"feature_name": "salary_transfer_freq_12m", "interpretation": "more regular salary inflow should indicate repayment stability"},
    {"feature_name": "cb_purchase_util_avg_3m", "interpretation": "higher recent purchase utilization indicates stronger repayment pressure"},
    {"feature_name": "ext_cb_default_flag", "interpretation": "external bureau default or severe bureau delinquency should materially increase risk"},
    {"feature_name": "acct_tenure_months", "interpretation": "longer relationship tenure should generally reduce risk"},
)


MODEL_RISK_NOTES: tuple[dict[str, str], ...] = (
    {"risk_topic": "leakage", "note": "all binning, WoE, and coefficient estimation must be fit on development sample only"},
    {"risk_topic": "sign_inconsistency", "note": "coefficients contradicting expected business direction require review or variable replacement"},
    {"risk_topic": "multicollinearity", "note": "high VIF among WoE variables can destabilize coefficients even when IV is strong"},
    {"risk_topic": "overfitting", "note": "keep final variable count around 10-12 and avoid tail-bin driven predictors"},
    {"risk_topic": "case_selection_bias", "note": "case comparison should weigh interpretability and stability, not discrimination only"},
)


def build_modeling_workflow() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in MODELING_WORKFLOW)


def build_candidate_model_cases() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in MODEL_CASES)


def build_final_variable_set() -> pd.DataFrame:
    return pd.DataFrame(FINAL_VARIABLE_SET)


def build_coefficient_interpretation() -> pd.DataFrame:
    return pd.DataFrame(COEFFICIENT_INTERPRETATION)


def build_model_risk_notes() -> pd.DataFrame:
    return pd.DataFrame(MODEL_RISK_NOTES)


def fit_scorecard_logit(
    woe_df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
) -> tuple[sm.Logit, sm.discrete.discrete_model.BinaryResultsWrapper]:
    """Fit a logistic regression on WoE-transformed variables."""
    _require_statsmodels()
    prepared_cols = _prepare_feature_columns(woe_df, feature_cols)
    x = sm.add_constant(woe_df[prepared_cols], has_constant="add")
    y = woe_df[target_col].astype(int)
    model = sm.Logit(y, x)
    try:
        result = model.fit(disp=False)
    except Exception:
        result = model.fit_regularized(disp=False)
    return model, result


def calculate_vif(woe_df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Calculate VIF without external dependencies."""
    _require_statsmodels()
    cols = _prepare_feature_columns(woe_df, feature_cols)
    rows: list[dict[str, float | str]] = []
    for col in cols:
        y = woe_df[col].astype(float)
        x_cols = [item for item in cols if item != col]
        if not x_cols:
            rows.append({"feature_name": col, "vif": 1.0})
            continue
        x = sm.add_constant(woe_df[x_cols], has_constant="add")
        r_squared = sm.OLS(y, x).fit().rsquared
        vif = np.inf if r_squared >= 0.999999 else 1.0 / (1.0 - r_squared)
        rows.append({"feature_name": col, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def evaluate_sign_consistency(
    coefficients: pd.Series,
    expected_sign_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare fitted signs to expected business directions."""
    if expected_sign_df is None:
        expected_sign_df = build_final_variable_set()
    expected_map = expected_sign_df.set_index("feature_name")["expected_sign"].to_dict()
    rows: list[dict[str, str | float]] = []
    for feature_name, coef in coefficients.items():
        if feature_name == "const":
            continue
        raw_feature_name = feature_name.replace("__woe", "")
        actual_sign = "+" if coef > 0 else "-" if coef < 0 else "0"
        raw_expected_sign = expected_map.get(raw_feature_name, "unknown")
        if feature_name.endswith("__woe") and raw_expected_sign in {"+", "-"}:
            expected_sign = "-"
        else:
            expected_sign = raw_expected_sign
        rows.append(
            {
                "feature_name": feature_name,
                "raw_feature_name": raw_feature_name,
                "coefficient": float(coef),
                "expected_sign": expected_sign,
                "actual_sign": actual_sign,
                "sign_match_flag": "Y" if expected_sign == actual_sign else "N",
            }
        )
    return pd.DataFrame(rows)


def summarize_model_diagnostics(
    result: sm.discrete.discrete_model.BinaryResultsWrapper,
    woe_df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Return p-value, coefficient, VIF, and sign-check tables."""
    used_feature_cols = [col for col in result.params.index if col != "const"]
    coef_df = pd.DataFrame(
        {
            "feature_name": result.params.index,
            "coefficient": result.params.values,
            "p_value": getattr(result, "pvalues", pd.Series(np.nan, index=result.params.index)).values,
        }
    )
    vif_df = calculate_vif(woe_df=woe_df, feature_cols=used_feature_cols)
    sign_df = evaluate_sign_consistency(result.params)
    return {
        "coefficients": coef_df,
        "vif": vif_df,
        "sign_check": sign_df,
    }


def compare_model_cases(
    woe_df: pd.DataFrame,
    target_col: str,
    case_feature_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Fit and compare multiple model cases using basic diagnostics."""
    _require_statsmodels()
    rows: list[dict[str, object]] = []
    for case_id, feature_cols in case_feature_map.items():
        _, result = fit_scorecard_logit(woe_df=woe_df, target_col=target_col, feature_cols=feature_cols)
        used_feature_cols = [col for col in result.params.index if col != "const"]
        vif_df = calculate_vif(woe_df=woe_df, feature_cols=used_feature_cols)
        sign_df = evaluate_sign_consistency(result.params)
        rows.append(
            {
                "case_id": case_id,
                "variable_count": len(used_feature_cols),
                "aic": float(result.aic),
                "max_p_value": float(
                    getattr(result, "pvalues", pd.Series(np.nan, index=result.params.index))
                    .drop("const", errors="ignore")
                    .max()
                ),
                "max_vif": float(vif_df["vif"].max()) if not vif_df.empty else np.nan,
                "sign_mismatch_count": int((sign_df["sign_match_flag"] == "N").sum()) if not sign_df.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["sign_mismatch_count", "max_p_value", "aic"])


def _require_statsmodels() -> None:
    if sm is None:
        raise ModuleNotFoundError(
            "statsmodels is required for logistic regression, p-value, and VIF diagnostics."
        )


def _prepare_feature_columns(
    woe_df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> list[str]:
    cols = [col for col in feature_cols if col in woe_df.columns]
    if not cols:
        return []
    x = woe_df[cols].copy()
    nunique = x.nunique(dropna=False)
    cols = [col for col in cols if nunique[col] > 1]
    if not cols:
        return []
    x = woe_df[cols].copy()
    duplicate_mask = x.T.duplicated()
    cols = [col for col in x.columns if not duplicate_mask.loc[col]]
    return cols


def fit_woe_binning(
    feature_df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    n_bins: int = 5,
) -> dict[str, dict[str, object]]:
    """Fit simple quantile-based WoE maps on the development sample."""
    maps: dict[str, dict[str, object]] = {}
    for feature_name in feature_cols:
        series = feature_df[feature_name]
        temp = pd.DataFrame({"x": series, "y": feature_df[target_col].astype(int)})
        temp["bin"] = "MISSING"
        non_missing = temp["x"].notna()
        edges = None
        labels = None
        if non_missing.sum() > 0:
            try:
                binned = pd.qcut(
                    temp.loc[non_missing, "x"],
                    q=min(n_bins, non_missing.sum()),
                    duplicates="drop",
                )
            except ValueError:
                binned = pd.cut(
                    temp.loc[non_missing, "x"],
                    bins=min(n_bins, max(2, temp.loc[non_missing, "x"].nunique())),
                    duplicates="drop",
                )
            labels = [str(cat) for cat in binned.cat.categories]
            edges = [binned.cat.categories[0].left] + [cat.right for cat in binned.cat.categories]
            temp.loc[non_missing, "bin"] = binned.astype(str)
        grouped = temp.groupby("bin").agg(total_cnt=("y", "size"), bad_cnt=("y", "sum"))
        grouped["good_cnt"] = grouped["total_cnt"] - grouped["bad_cnt"]
        bad_total = grouped["bad_cnt"].sum()
        good_total = grouped["good_cnt"].sum()
        grouped["bad_dist"] = grouped["bad_cnt"].clip(lower=0.5) / max(bad_total, 1)
        grouped["good_dist"] = grouped["good_cnt"].clip(lower=0.5) / max(good_total, 1)
        grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
        grouped["iv_component"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]
        grouped = grouped.reset_index()
        maps[feature_name] = {"map_df": grouped, "edges": edges, "labels": labels}
    return maps


def transform_to_woe(
    feature_df: pd.DataFrame,
    woe_maps: dict[str, dict[str, object]],
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """Apply WoE maps to a feature dataset."""
    out = feature_df.copy()
    for feature_name in feature_cols:
        series = out[feature_name]
        binned = pd.Series("MISSING", index=out.index, dtype=object)
        non_missing = series.notna()
        if non_missing.sum() > 0:
            edges = woe_maps[feature_name]["edges"]
            labels = woe_maps[feature_name]["labels"]
            if edges is not None and labels is not None and len(labels) > 0:
                cut_result = pd.cut(
                    series.loc[non_missing],
                    bins=edges,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                )
                binned.loc[non_missing] = cut_result.astype(str)
        map_df = woe_maps[feature_name]["map_df"][["bin", "woe"]].copy()
        out[f"{feature_name}__woe"] = binned.map(map_df.set_index("bin")["woe"]).fillna(0.0)
    return out


def assign_woe_bins(
    feature_df: pd.DataFrame,
    woe_maps: dict[str, dict[str, object]],
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """Assign development-fitted bins to each row."""
    out = feature_df.copy()
    for feature_name in feature_cols:
        series = out[feature_name]
        binned = pd.Series("MISSING", index=out.index, dtype=object)
        non_missing = series.notna()
        if non_missing.sum() > 0:
            edges = woe_maps[feature_name]["edges"]
            labels = woe_maps[feature_name]["labels"]
            if edges is not None and labels is not None and len(labels) > 0:
                cut_result = pd.cut(
                    series.loc[non_missing],
                    bins=edges,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                )
                binned.loc[non_missing] = cut_result.astype(str)
        out[f"{feature_name}__bin"] = binned
    return out


def run_proxy_scorecard_case_analysis(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    key_cols: tuple[str, str] = ("고객번호", "기준년월"),
    target_col: str = "bad_flag",
) -> dict[str, pd.DataFrame]:
    """Run provisional WoE-logit case analysis using the proxy label."""
    final_vars = build_final_variable_set()["feature_name"].tolist()
    merged = feature_df.merge(
        label_df[list(key_cols) + [target_col]],
        on=list(key_cols),
        how="inner",
        validate="one_to_one",
    )
    val_df = merged.loc[merged["기준년월"] == 202209].copy()
    dev_df = merged.loc[merged["기준년월"] == 202309].copy()
    oot_df = merged.loc[merged["기준년월"] == 202409].copy()

    case_feature_map = {
        "C1": final_vars[:10],
        "C2": final_vars[:11],
        "C3": final_vars[:11] + ["loan_exposure_total_bal"][: max(0, 12 - len(final_vars[:11]))],
    }

    woe_maps = fit_woe_binning(dev_df, target_col=target_col, feature_cols=set(sum(case_feature_map.values(), [])))
    dev_woe = transform_to_woe(dev_df, woe_maps=woe_maps, feature_cols=woe_maps.keys())
    val_woe = transform_to_woe(val_df, woe_maps=woe_maps, feature_cols=woe_maps.keys())
    oot_woe = transform_to_woe(oot_df, woe_maps=woe_maps, feature_cols=woe_maps.keys())

    case_input = {
        case_id: [f"{name}__woe" for name in feature_list]
        for case_id, feature_list in case_feature_map.items()
    }
    case_compare_df = compare_model_cases(dev_woe, target_col=target_col, case_feature_map=case_input)
    best_case_id = case_compare_df.iloc[0]["case_id"]
    best_features = case_input[str(best_case_id)]
    _, result = fit_scorecard_logit(dev_woe, target_col=target_col, feature_cols=best_features)
    diagnostics = summarize_model_diagnostics(result, dev_woe, best_features)

    return {
        "dev_sample": dev_df,
        "validation_sample": val_df,
        "oot_sample": oot_df,
        "woe_maps": woe_maps,
        "woe_transformed_dev": dev_woe,
        "woe_transformed_validation": val_woe,
        "woe_transformed_oot": oot_woe,
        "best_case_id": pd.DataFrame({"best_case_id": [best_case_id]}),
        "best_features": pd.DataFrame({"feature_name": [name.replace("__woe", "") for name in best_features]}),
        "case_comparison": case_compare_df,
        "coefficients": diagnostics["coefficients"],
        "vif": diagnostics["vif"],
        "sign_check": diagnostics["sign_check"],
    }


def build_provisional_scorecard_table(
    woe_maps: dict[str, pd.DataFrame],
    coefficients_df: pd.DataFrame,
    pdo: float = 40.0,
) -> pd.DataFrame:
    """Build a provisional variable-bin score table from WoE bins and fitted coefficients."""
    factor = pdo / np.log(2.0)
    coef_map = (
        coefficients_df.loc[coefficients_df["feature_name"] != "const", ["feature_name", "coefficient"]]
        .assign(raw_feature_name=lambda df: df["feature_name"].str.replace("__woe", "", regex=False))
        .set_index("raw_feature_name")["coefficient"]
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for feature_name, map_df in woe_maps.items():
        coefficient = coef_map.get(feature_name)
        if coefficient is None:
            continue
        temp = map_df["map_df"].copy()
        temp["feature_name"] = feature_name
        temp["coefficient"] = coefficient
        temp["bin_points"] = -(coefficient * temp["woe"] * factor)
        rows.extend(temp.to_dict("records"))
    return pd.DataFrame(rows)[
        [
            "feature_name",
            "bin",
            "total_cnt",
            "bad_cnt",
            "good_cnt",
            "woe",
            "iv_component",
            "coefficient",
            "bin_points",
        ]
    ]
