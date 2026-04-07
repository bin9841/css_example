from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelCase:
    case_name: str
    objective: str
    design_notes: str
    variable_count_target: str


@dataclass(frozen=True)
class ModelVariable:
    feature_name: str
    information_domain: str
    expected_sign: str
    business_rationale: str


MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase(
        "conservative",
        "maximize stability and interpretability",
        "favor the strongest borrower-behavior anchors only; keep redundancy low",
        "10 variables",
    ),
    ModelCase(
        "balanced",
        "balance discrimination and business coverage",
        "keep one representative variable from each major domain plus one secondary behavioral signal where needed",
        "11 variables",
    ),
    ModelCase(
        "coverage",
        "maximize domain coverage while staying scorecard-friendly",
        "retain business-meaningful secondary variables if VIF and p-values remain acceptable",
        "12 variables",
    ),
)


FINAL_VARIABLE_SET: tuple[ModelVariable, ...] = (
    ModelVariable("int_dq_days_max_12m", "internal_delinquency", "+", "strongest direct borrower stress indicator"),
    ModelVariable("int_dq_recency_3m_flag", "internal_delinquency", "+", "recent arrears improves short-term sensitivity"),
    ModelVariable("deposit_liquidity_share_3m", "deposit_performance", "-", "liquidity depth is a stable relationship-quality proxy"),
    ModelVariable("salary_transfer_freq_12m", "salary_transfer", "-", "regular salary transfer indicates income stability"),
    ModelVariable("salary_transfer_avg_amt_6m", "salary_transfer", "-", "salary amount captures income scale and stability"),
    ModelVariable("cb_card_util_curr", "cb_card_usage", "+", "high utilization signals repayment pressure"),
    ModelVariable("cb_cash_service_freq_6m", "cb_card_usage", "+", "cash-service dependence often precedes stress"),
    ModelVariable("ext_cb_default_flag", "external_delinquency", "+", "bureau default registration is a strong distress flag"),
    ModelVariable("loan_exposure_nonbank_share", "loan_exposure", "+", "high non-bank share indicates funding pressure"),
    ModelVariable("loan_new_lender_freq_12m", "loan_exposure", "+", "frequent new borrowing points to growing burden"),
    ModelVariable("acct_tenure_months", "account_opening_history", "-", "longer tenure usually reflects stronger relationship depth"),
)


def build_modeling_workflow() -> pd.DataFrame:
    """Return the scorecard modeling workflow."""
    return pd.DataFrame(
        [
            {
                "step_no": 1,
                "step_name": "woe_binning",
                "purpose": "convert screened raw-derived variables into scorecard-ready WoE bins",
                "controls": "freeze cut points on dev sample only; keep special bins explicit",
            },
            {
                "step_no": 2,
                "step_name": "logistic_fit",
                "purpose": "fit an interpretable logistic regression on WoE-transformed variables",
                "controls": "use dev-only estimation; review p-values, VIF, and sign consistency",
            },
            {
                "step_no": 3,
                "step_name": "case_analysis",
                "purpose": "compare conservative, balanced, and coverage-driven model cases",
                "controls": "retain business interpretable variables and remove unstable redundancy",
            },
            {
                "step_no": 4,
                "step_name": "scorecard_translation",
                "purpose": "map coefficients to score points with a transparent scaling rule",
                "controls": "preserve monotonicity and document points-per-WoE mapping",
            },
            {
                "step_no": 5,
                "step_name": "validation_ready_checks",
                "purpose": "prepare coefficient interpretation and risk notes for validation",
                "controls": "report sign checks, p-values, VIF, and sample stability",
            },
        ]
    )


def build_candidate_model_cases() -> pd.DataFrame:
    """Return the candidate modeling cases."""
    return pd.DataFrame(asdict(item) for item in MODEL_CASES)


def build_final_variable_set() -> pd.DataFrame:
    """Return the recommended final variable set."""
    return pd.DataFrame(asdict(item) for item in FINAL_VARIABLE_SET)


def build_coefficient_interpretation() -> pd.DataFrame:
    """Return the expected coefficient direction and business meaning."""
    rows = []
    for item in FINAL_VARIABLE_SET:
        rows.append(
            {
                "feature_name": item.feature_name,
                "expected_sign": item.expected_sign,
                "interpretation": item.business_rationale,
            }
        )
    return pd.DataFrame(rows)


def build_model_risk_notes() -> pd.DataFrame:
    """Return the main scorecard modeling risks."""
    return pd.DataFrame(
        [
            {
                "risk": "leakage",
                "note": "Any feature using post-as-of or performance-window data will invalidate the scorecard.",
            },
            {
                "risk": "multicollinearity",
                "note": "Closely related behavioral variables can inflate VIF and destabilize coefficients.",
            },
            {
                "risk": "sign_instability",
                "note": "Variables with unstable WoE binning can flip coefficient sign across samples.",
            },
            {
                "risk": "p_value_instability",
                "note": "Small or sparse bins can produce weak p-values and overfit effects.",
            },
            {
                "risk": "business_mismatch",
                "note": "A variable can be statistically strong but still hard to explain in review or operations.",
            },
        ]
    )


def prepare_woe_input(
    feature_df: pd.DataFrame,
    woe_map_df: pd.DataFrame,
    key_cols: tuple[str, str] = ("고객번호", "기준년월"),
) -> pd.DataFrame:
    """Apply WoE mappings to a feature table."""
    out = feature_df[list(key_cols)].copy()
    for feature_name in woe_map_df["feature_name"].unique():
        feature_map = woe_map_df.loc[woe_map_df["feature_name"] == feature_name, ["bin_label", "woe"]]
        value_col = f"{feature_name}__bin"
        if value_col not in feature_df.columns:
            continue
        merged = feature_df[[*key_cols, value_col]].merge(
            feature_map,
            left_on=value_col,
            right_on="bin_label",
            how="left",
        )
        out[f"{feature_name}__woe"] = merged["woe"].fillna(0.0).values
    return out


def calculate_vif_matrix(x_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for a modeling design matrix."""
    numeric_df = x_df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if numeric_df.empty:
        return pd.DataFrame(columns=["variable", "vif"])
    vif_rows = []
    values = numeric_df.to_numpy(dtype=float)
    for idx, column in enumerate(numeric_df.columns):
        y = values[:, idx]
        x = np.delete(values, idx, axis=1)
        if x.size == 0:
            vif = 1.0
        else:
            x = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            y_hat = x @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1 - ss_res / ss_tot))
            vif = float(1.0 / max(1e-12, 1.0 - r_squared))
        vif_rows.append({"variable": column, "vif": vif})
    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False)


def fit_logistic_scorecard(x_df: pd.DataFrame, y: pd.Series):
    """Fit a logistic regression scorecard model on WoE features.

    Uses statsmodels if available so p-values can be reported. If statsmodels is not
    installed, raise a clear error and keep the rest of the module usable.
    """
    try:
        import statsmodels.api as sm  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "statsmodels is required for p-value-enabled scorecard fitting."
        ) from exc

    x = sm.add_constant(x_df, has_constant="add")
    model = sm.Logit(y.astype(int), x)
    return model.fit(disp=False)


def summarize_coefficients(model) -> pd.DataFrame:
    """Return coefficients, p-values, and sign diagnostics."""
    params = model.params
    pvalues = model.pvalues
    summary = pd.DataFrame(
        {
            "variable": params.index,
            "coefficient": params.values,
            "p_value": pvalues.values,
        }
    )
    summary["sign"] = np.where(summary["coefficient"] >= 0, "+", "-")
    return summary
