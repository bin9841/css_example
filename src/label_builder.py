from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.public_aliases import (
    ALIAS_EXTERNAL_BUREAU_DEFAULT_FLAG_12M,
    ALIAS_EXTERNAL_BUREAU_SEVERE_DQ_60PLUS_FLAG_12M,
    apply_public_column_aliases,
)


@dataclass(frozen=True)
class LabelingRequirement:
    requirement_name: str
    description: str


LABELING_REQUIREMENTS: tuple[LabelingRequirement, ...] = (
    LabelingRequirement(
        "forward_performance_window",
        "Need borrower-level events observed from the month after reference_month through the next 12 months.",
    ),
    LabelingRequirement(
        "borrower_level_event_timeline",
        "Need internal and external event dates aggregated to borrower level.",
    ),
    LabelingRequirement(
        "event_types",
        "Need >=60 DPD, default, charge-off, and external bureau default registration events.",
    ),
    LabelingRequirement(
        "optional_indeterminate",
        "Need 10-59 DPD event history if indeterminate labeling is to be operationalized.",
    ),
)


def build_labeling_requirements() -> pd.DataFrame:
    """Return the required inputs for borrower-level performance labeling."""
    return pd.DataFrame(asdict(item) for item in LABELING_REQUIREMENTS)


def build_label_table_schema() -> pd.DataFrame:
    """Return the target label-table schema."""
    return pd.DataFrame(
        [
            {"field_name": "고객번호", "description": "borrower identifier"},
            {"field_name": "기준년월", "description": "observation month"},
            {"field_name": "bad_flag", "description": "borrower-level bad flag over 12M performance period"},
            {"field_name": "indeterminate_flag", "description": "optional 10-59 DPD gray-zone flag"},
            {"field_name": "bad_event_type", "description": "internal_60dpd, external_60dpd, default_event, or none"},
            {"field_name": "bad_event_source", "description": "internal, external, or both"},
            {"field_name": "bad_event_date", "description": "earliest qualifying bad event date"},
            {"field_name": "perf_start_month", "description": "month after 기준년월"},
            {"field_name": "perf_end_month", "description": "12-month forward performance end month"},
        ]
    )


def build_label_table_from_snapshot(*args, **kwargs) -> pd.DataFrame:
    """Raise a clear blocker because the current snapshot dataset lacks forward performance events."""
    raise NotImplementedError(
        "Current workspace dataset contains backward-looking 12M history but does not provide "
        "forward 12M borrower-level performance events required to build a Basel-style bad_flag. "
        "Provide a forward event table or approve a temporary proxy-label approach."
    )


def build_proxy_label_table_from_snapshot(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a temporary proxy label table from current-snapshot signals only.

    This is for demonstration and pipeline testing only. It is not a valid
    Basel-style forward performance label.
    """
    col_customer = "고객번호"
    col_ref_month = "기준년월"
    col_curr_dq = "현재연체여부"
    col_internal_max_dpd = "내부카드최장연체일수_12M"
    col_external_max_dpd = "외부최장연체일수_12M"
    base_df = apply_public_column_aliases(base_df)
    col_external_bureau_default = ALIAS_EXTERNAL_BUREAU_DEFAULT_FLAG_12M
    col_external_bureau_severe_dq = ALIAS_EXTERNAL_BUREAU_SEVERE_DQ_60PLUS_FLAG_12M

    required_cols = [
        col_customer,
        col_ref_month,
        col_curr_dq,
        col_internal_max_dpd,
        col_external_max_dpd,
        col_external_bureau_default,
        col_external_bureau_severe_dq,
    ]
    missing_cols = [col for col in required_cols if col not in base_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required proxy-label columns: {missing_cols}")

    label_df = base_df[[col_customer, col_ref_month]].copy()
    internal_bad = base_df[col_internal_max_dpd].fillna(0) >= 60
    external_bad = base_df[col_external_max_dpd].fillna(0) >= 60
    bureau_default = base_df[[col_external_bureau_default, col_external_bureau_severe_dq]].fillna(0).max(axis=1) > 0
    current_bad = base_df[col_curr_dq].fillna(0) > 0

    label_df["bad_flag"] = (
        internal_bad | external_bad | bureau_default | current_bad
    ).astype(int)
    label_df["indeterminate_flag"] = (
        (
            base_df[col_internal_max_dpd].fillna(0).between(10, 59)
            | base_df[col_external_max_dpd].fillna(0).between(10, 59)
        )
        & (label_df["bad_flag"] == 0)
    ).astype(int)
    label_df["bad_event_type"] = "none"
    label_df.loc[current_bad, "bad_event_type"] = "current_snapshot_delinquency"
    label_df.loc[internal_bad, "bad_event_type"] = "internal_60dpd_history_proxy"
    label_df.loc[external_bad, "bad_event_type"] = "external_60dpd_history_proxy"
    label_df.loc[bureau_default, "bad_event_type"] = "bureau_default_proxy"
    label_df["bad_event_source"] = "proxy_snapshot"
    label_df["bad_event_date"] = pd.NA
    label_df["perf_start_month"] = pd.NA
    label_df["perf_end_month"] = pd.NA
    label_df["label_status"] = "good"
    label_df.loc[label_df["indeterminate_flag"] == 1, "label_status"] = "indeterminate"
    label_df.loc[label_df["bad_flag"] == 1, "label_status"] = "bad"
    label_df["label_definition_type"] = "proxy_snapshot"
    return label_df
