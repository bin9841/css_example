from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.candidate_features import FEATURE_SPECS


@dataclass(frozen=True)
class ScreeningRule:
    stage: str
    rule_name: str
    threshold: str
    action: str


SCREENING_RULES: tuple[ScreeningRule, ...] = (
    ScreeningRule(
        "fine_classing",
        "minimum_bin_support",
        ">= 5% sample share per interior bin, with special bins allowed",
        "merge sparse bins before IV evaluation",
    ),
    ScreeningRule(
        "iv_screening",
        "iv_floor",
        "IV < 0.10",
        "drop unless clearly domain-critical or operationally required",
    ),
    ScreeningRule(
        "iv_screening",
        "iv_keep_zone",
        "0.10 <= IV < 0.30",
        "keep as preferred shortlist candidate if stable and interpretable",
    ),
    ScreeningRule(
        "iv_screening",
        "iv_preferred_zone",
        "IV >= 0.30",
        "review for leakage and saturation before keeping",
    ),
    ScreeningRule(
        "stability_screening",
        "psi_threshold",
        "PSI <= 0.01",
        "keep only if variable distribution is stable between development and validation periods",
    ),
    ScreeningRule(
        "correlation_filtering",
        "high_corr_cluster",
        "|corr| >= 0.70 on dev sample (prefer Spearman for monotonic scorecard signals)",
        "retain a single representative variable per cluster",
    ),
    ScreeningRule(
        "coarse_classing",
        "woe_monotonicity",
        "bins should show monotonic or near-monotonic WoE",
        "merge adjacent bins until pattern is stable",
    ),
)


DOMAIN_VARIABLE_PRIORITY: tuple[dict[str, object], ...] = (
    {
        "information_domain": "internal_delinquency",
        "candidate_variables": (
            "int_dq_days_max_12m",
            "int_dq_months_freq_12m",
            "int_dq_amt_sum_6m",
            "int_dq_recency_3m_flag",
            "int_dq_trend_last3_vs_prev3",
            "int_dq_volatility_6m",
        ),
        "selection_rule": "prefer max severity first, then frequency, then trend/volatility if IV and stability support it",
    },
    {
        "information_domain": "deposit_performance",
        "candidate_variables": (
            "deposit_avg_total_bal_3m",
            "deposit_avg_demand_bal_6m",
            "deposit_trend_total_bal_3m_vs_6m",
            "deposit_volatility_total_bal_6m",
            "deposit_liquidity_share_3m",
        ),
        "selection_rule": "prefer liquidity/stability and trend variables with low missingness",
    },
    {
        "information_domain": "salary_transfer",
        "candidate_variables": (
            "salary_transfer_freq_12m",
            "salary_transfer_avg_amt_6m",
            "salary_transfer_recency_3m_flag",
            "salary_transfer_volatility_6m",
            "salary_transfer_trend_3m_vs_6m",
        ),
        "selection_rule": "prefer recency/frequency first, then amount stability and trend",
    },
    {
        "information_domain": "cb_card_usage",
        "candidate_variables": (
            "cb_card_util_curr",
            "cb_purchase_util_avg_3m",
            "cb_cash_service_freq_6m",
            "cb_card_loan_util_trend_3m_vs_6m",
            "cb_util_volatility_6m",
        ),
        "selection_rule": "prefer current utilization plus one behavioral trend variable",
    },
    {
        "information_domain": "external_delinquency",
        "candidate_variables": (
            "ext_dq_max_days_12m",
            "ext_dq_freq_12m",
            "ext_dq_recency_3m_flag",
            "ext_cb_default_flag",
        ),
        "selection_rule": "prefer max severity or default flag; keep one external distress measure",
    },
    {
        "information_domain": "loan_exposure",
        "candidate_variables": (
            "loan_exposure_total_cnt",
            "loan_exposure_total_bal",
            "loan_exposure_nonbank_share",
            "loan_new_lender_freq_12m",
            "loan_recent_new_lender_trend_3m_6m",
        ),
        "selection_rule": "prefer one balance measure and one breadth/pressure measure",
    },
    {
        "information_domain": "account_opening_history",
        "candidate_variables": (
            "acct_tenure_months",
            "acct_card_tenure_months",
            "acct_recent_opening_freq_24m",
            "acct_recent_opening_intensity_12m_24m",
        ),
        "selection_rule": "prefer tenure plus recent opening intensity",
    },
)


SHORTLISTED_VARIABLE_POOL: tuple[str, ...] = (
    "int_dq_days_max_12m",
    "int_dq_months_freq_12m",
    "int_dq_recency_3m_flag",
    "int_dq_trend_last3_vs_prev3",
    "deposit_avg_total_bal_3m",
    "deposit_trend_total_bal_3m_vs_6m",
    "deposit_liquidity_share_3m",
    "salary_transfer_freq_12m",
    "salary_transfer_avg_amt_6m",
    "salary_transfer_recency_3m_flag",
    "cb_card_util_curr",
    "cb_purchase_util_avg_3m",
    "cb_cash_service_freq_6m",
    "ext_dq_max_days_12m",
    "ext_cb_default_flag",
    "loan_exposure_total_bal",
    "loan_exposure_nonbank_share",
    "loan_new_lender_freq_12m",
    "acct_tenure_months",
    "acct_recent_opening_freq_24m",
)


CORRELATION_GROUPS: tuple[dict[str, object], ...] = (
    {
        "group_id": "D1",
        "information_domain": "internal_delinquency",
        "cluster_rule": "keep the strongest severity variable and one recency variable",
        "members": (
            "int_dq_days_max_12m",
            "int_dq_months_freq_12m",
            "int_dq_recency_3m_flag",
            "int_dq_trend_last3_vs_prev3",
            "int_dq_volatility_6m",
        ),
        "representative": "int_dq_days_max_12m",
    },
    {
        "group_id": "D2",
        "information_domain": "deposit_performance",
        "cluster_rule": "keep one level variable and one trend/liquidity variable",
        "members": (
            "deposit_avg_total_bal_3m",
            "deposit_avg_demand_bal_6m",
            "deposit_trend_total_bal_3m_vs_6m",
            "deposit_volatility_total_bal_6m",
            "deposit_liquidity_share_3m",
        ),
        "representative": "deposit_liquidity_share_3m",
    },
    {
        "group_id": "D3",
        "information_domain": "salary_transfer",
        "cluster_rule": "keep one recency/frequency measure and one amount measure",
        "members": (
            "salary_transfer_freq_12m",
            "salary_transfer_avg_amt_6m",
            "salary_transfer_recency_3m_flag",
            "salary_transfer_volatility_6m",
            "salary_transfer_trend_3m_vs_6m",
        ),
        "representative": "salary_transfer_freq_12m",
    },
    {
        "group_id": "D4",
        "information_domain": "cb_card_usage",
        "cluster_rule": "keep one utilization measure and one pressure measure",
        "members": (
            "cb_card_util_curr",
            "cb_purchase_util_avg_3m",
            "cb_cash_service_freq_6m",
            "cb_card_loan_util_trend_3m_vs_6m",
            "cb_util_volatility_6m",
        ),
        "representative": "cb_card_util_curr",
    },
    {
        "group_id": "D5",
        "information_domain": "external_delinquency",
        "cluster_rule": "keep one external distress severity variable",
        "members": (
            "ext_dq_max_days_12m",
            "ext_dq_freq_12m",
            "ext_dq_recency_3m_flag",
            "ext_cb_default_flag",
        ),
        "representative": "ext_cb_default_flag",
    },
    {
        "group_id": "D6",
        "information_domain": "loan_exposure",
        "cluster_rule": "keep one balance variable and one breadth variable",
        "members": (
            "loan_exposure_total_cnt",
            "loan_exposure_total_bal",
            "loan_exposure_nonbank_share",
            "loan_new_lender_freq_12m",
            "loan_recent_new_lender_trend_3m_6m",
        ),
        "representative": "loan_exposure_nonbank_share",
    },
    {
        "group_id": "D7",
        "information_domain": "account_opening_history",
        "cluster_rule": "keep tenure plus recent opening intensity",
        "members": (
            "acct_tenure_months",
            "acct_card_tenure_months",
            "acct_recent_opening_freq_24m",
            "acct_recent_opening_intensity_12m_24m",
        ),
        "representative": "acct_tenure_months",
    },
)


REPRESENTATIVE_VARIABLES: tuple[str, ...] = (
    "int_dq_days_max_12m",
    "int_dq_recency_3m_flag",
    "deposit_liquidity_share_3m",
    "salary_transfer_freq_12m",
    "salary_transfer_avg_amt_6m",
    "cb_card_util_curr",
    "cb_cash_service_freq_6m",
    "ext_cb_default_flag",
    "loan_exposure_nonbank_share",
    "loan_new_lender_freq_12m",
    "acct_tenure_months",
)


CLASSING_GUIDANCE: tuple[dict[str, object], ...] = (
    {
        "topic": "fine_classing",
        "guidance": "Start with many small bins, then merge until each interior bin has enough sample and bad counts.",
        "practical_rule": "Use separate bins for missing, zero, structural zero, and top-code if they behave differently.",
    },
    {
        "topic": "woe_monotonicity",
        "guidance": "Prefer monotonic or near-monotonic WoE across bins for scorecard stability.",
        "practical_rule": "If WoE flips direction, merge adjacent bins or revisit the variable.",
    },
    {
        "topic": "special_bins",
        "guidance": "Keep special bins for missing, no history, and zero exposure when these states are business-meaningful.",
        "practical_rule": "Do not force special states into regular bins just to improve smoothness.",
    },
    {
        "topic": "coarse_classing",
        "guidance": "After fine classing, collapse bins into a small set suitable for WoE and logistic regression.",
        "practical_rule": "Target 4-8 coarse bins for most variables, fewer for sparse or highly skewed signals.",
    },
    {
        "topic": "bin_stability",
        "guidance": "Each bin should have enough observations and enough bads for stable WoE estimates.",
        "practical_rule": "Merge tail bins first when counts are small or IV is driven by unstable tails.",
    },
    {
        "topic": "leakage_control",
        "guidance": "No bin cut should use information from validation or OOT samples.",
        "practical_rule": "Determine bins on development sample only and freeze them for all splits.",
    },
)


def build_shortlisted_variable_pool() -> pd.DataFrame:
    """Return the short-listed variable pool after initial screening."""
    return pd.DataFrame(
        {
            "feature_name": spec.feature_name,
            "feature_group": spec.feature_group,
            "pattern_type": spec.pattern_type,
            "iv_target_threshold": 0.10,
            "psi_target_threshold": 0.01,
        }
        for spec in FEATURE_SPECS
        if spec.feature_name in SHORTLISTED_VARIABLE_POOL
    )


def build_grouped_variable_table() -> pd.DataFrame:
    """Return the domain grouping table for correlation screening."""
    return pd.DataFrame(CORRELATION_GROUPS)


def build_representative_variable_list() -> pd.DataFrame:
    """Return the representative variable list by information domain."""
    return pd.DataFrame(
        {
            "feature_name": name,
            "selected": 1,
        }
        for name in REPRESENTATIVE_VARIABLES
    )


def build_classing_guidance() -> pd.DataFrame:
    """Return the coarse classing guidance catalog."""
    return pd.DataFrame(CLASSING_GUIDANCE)


def build_variable_screening_workflow() -> pd.DataFrame:
    """Return the end-to-end screening workflow."""
    return pd.DataFrame(
        [
            {
                "step_no": 1,
                "step_name": "fine_classing",
                "purpose": "define initial bins with enough support and interpretable risk direction",
                "output": "fine classing table",
            },
            {
                "step_no": 2,
                "step_name": "iv_screening",
                "purpose": "remove weak or unstable variables using IV thresholds and business review",
                "output": "IV-ranked shortlist",
            },
            {
                "step_no": 3,
                "step_name": "correlation_filtering",
                "purpose": "group highly correlated variables and keep one representative per cluster",
                "output": "correlation clusters",
            },
            {
                "step_no": 4,
                "step_name": "coarse_classing",
                "purpose": "collapse fine bins into stable WoE bins for modeling",
                "output": "final binning map",
            },
        ]
    )


def fine_classing(
    feature_df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create fine classes using quantile bins on the development sample."""
    outputs: list[pd.DataFrame] = []
    for feature_name in feature_cols:
        series = feature_df[feature_name]
        valid_mask = series.notna()
        if valid_mask.sum() < n_bins:
            continue
        try:
            binned = pd.qcut(series[valid_mask], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        temp = pd.DataFrame(
            {
                "feature_name": feature_name,
                "bin": binned.astype(str),
                "target": feature_df.loc[valid_mask, target_col].astype(int),
            }
        )
        grouped = temp.groupby(["feature_name", "bin"]).agg(
            total_cnt=("target", "size"),
            bad_cnt=("target", "sum"),
        )
        grouped["good_cnt"] = grouped["total_cnt"] - grouped["bad_cnt"]
        outputs.append(grouped.reset_index())
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def calculate_iv(fine_class_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate information value from fine-classing output."""
    if fine_class_df.empty:
        return pd.DataFrame(columns=["feature_name", "iv"])
    rows: list[dict[str, float | str]] = []
    for feature_name, part in fine_class_df.groupby("feature_name"):
        bad_total = part["bad_cnt"].sum()
        good_total = part["good_cnt"].sum()
        temp = part.copy()
        temp["bad_dist"] = temp["bad_cnt"].clip(lower=0.5) / max(bad_total, 1)
        temp["good_dist"] = temp["good_cnt"].clip(lower=0.5) / max(good_total, 1)
        temp["woe"] = np.log(temp["good_dist"] / temp["bad_dist"])
        temp["iv_component"] = (temp["good_dist"] - temp["bad_dist"]) * temp["woe"]
        rows.append({"feature_name": feature_name, "iv": float(temp["iv_component"].sum())})
    return pd.DataFrame(rows).sort_values("iv", ascending=False)


def filter_by_iv(iv_df: pd.DataFrame, min_iv: float = 0.02) -> pd.DataFrame:
    """Keep variables above the minimum IV threshold."""
    return iv_df.loc[iv_df["iv"] >= min_iv].copy()


def build_correlation_groups(
    feature_df: pd.DataFrame,
    feature_cols: Iterable[str],
    threshold: float = 0.70,
) -> pd.DataFrame:
    """Build simple correlation clusters for shortlisted variables."""
    numeric_df = feature_df[list(feature_cols)].select_dtypes(include=["number"])
    corr = numeric_df.corr(method="spearman").abs()
    groups: list[dict[str, object]] = []
    visited: set[str] = set()
    for anchor in corr.columns:
        if anchor in visited:
            continue
        partners = sorted(
            [
                other
                for other in corr.columns
                if other != anchor and corr.loc[anchor, other] >= threshold
            ]
        )
        visited.update([anchor, *partners])
        groups.append(
            {
                "group_anchor": anchor,
                "correlated_variables": ", ".join([anchor, *partners]),
                "group_size": 1 + len(partners),
            }
        )
    return pd.DataFrame(groups)


def run_variable_screening(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    target_col: str = "bad_flag",
    key_cols: tuple[str, str] = ("고객번호", "기준년월"),
    n_bins: int = 10,
    min_iv: float = 0.02,
    corr_threshold: float = 0.70,
) -> dict[str, pd.DataFrame]:
    """Run the end-to-end screening workflow on a labeled development sample."""
    merged_df = feature_df.merge(
        label_df[list(key_cols) + [target_col]],
        on=list(key_cols),
        how="inner",
        validate="one_to_one",
    )
    candidate_cols = [
        col for col in merged_df.columns if col not in [*key_cols, target_col]
    ]
    fine_class_df = fine_classing(
        feature_df=merged_df,
        target_col=target_col,
        feature_cols=candidate_cols,
        n_bins=n_bins,
    )
    iv_df = calculate_iv(fine_class_df)
    iv_filtered_df = filter_by_iv(iv_df, min_iv=min_iv)
    retained_cols = iv_filtered_df["feature_name"].tolist()
    correlation_df = build_correlation_groups(
        feature_df=merged_df,
        feature_cols=retained_cols,
        threshold=corr_threshold,
    )
    representative_df = _select_representatives(
        iv_df=iv_filtered_df,
        correlation_df=correlation_df,
    )
    return {
        "screening_input": merged_df,
        "fine_classing": fine_class_df,
        "iv_summary": iv_df,
        "iv_filtered": iv_filtered_df,
        "correlation_groups": correlation_df,
        "representative_variables": representative_df,
    }


def _select_representatives(
    iv_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Select one representative variable per correlation cluster using IV rank."""
    iv_rank = iv_df.set_index("feature_name")["iv"].to_dict()
    selected_rows: list[dict[str, object]] = []
    for _, row in correlation_df.iterrows():
        members = [item.strip() for item in str(row["correlated_variables"]).split(",")]
        ranked_members = sorted(
            members,
            key=lambda name: iv_rank.get(name, float("-inf")),
            reverse=True,
        )
        selected_rows.append(
            {
                "group_anchor": row["group_anchor"],
                "selected_feature": ranked_members[0] if ranked_members else None,
                "candidate_members": row["correlated_variables"],
                "selected_iv": iv_rank.get(ranked_members[0], np.nan)
                if ranked_members
                else np.nan,
            }
        )
    return pd.DataFrame(selected_rows)
