from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


BAD_DPD_THRESHOLD = 60
INDETERMINATE_DPD_LOWER = 10
INDETERMINATE_DPD_UPPER = 59
PERFORMANCE_WINDOW_MONTHS = 12


@dataclass(frozen=True)
class BadRule:
    rule_code: str
    event_scope: str
    threshold_or_event: str
    classification: str
    borrower_level_recognition: str


FINAL_BAD_RULES: tuple[BadRule, ...] = (
    BadRule(
        "internal_60dpd",
        "internal",
        "delinquency >= 60 DPD",
        "bad",
        "earliest qualifying internal account event defines borrower bad",
    ),
    BadRule(
        "external_60dpd",
        "external",
        "delinquency >= 60 DPD",
        "bad",
        "earliest qualifying external event defines borrower bad",
    ),
    BadRule(
        "default_event",
        "internal_or_external",
        "default / charge-off / external bureau default registration",
        "bad",
        "earliest qualifying default-type event defines borrower bad",
    ),
)


def build_bad_definition_table() -> pd.DataFrame:
    """Return the final borrower-level bad definition."""
    return pd.DataFrame(asdict(rule) for rule in FINAL_BAD_RULES)


def build_indeterminate_definition_table() -> pd.DataFrame:
    """Return the optional indeterminate definition."""
    return pd.DataFrame(
        [
            {
                "classification": "good",
                "condition": "0-9 DPD and no bad event in full 12M performance window",
                "recommended_treatment": "keep in binary development sample",
            },
            {
                "classification": "indeterminate",
                "condition": "10-59 DPD and no qualifying bad event",
                "recommended_treatment": "exclude from development or monitor separately",
            },
            {
                "classification": "bad",
                "condition": ">= 60 DPD or default / charge-off / external bureau default registration",
                "recommended_treatment": "keep as bad",
            },
        ]
    )


def build_label_structure() -> pd.DataFrame:
    """Return the expected label-table schema."""
    return pd.DataFrame(
        [
            {"field_name": "customer_id", "description": "borrower identifier"},
            {"field_name": "as_of_date", "description": "observation month-end"},
            {"field_name": "perf_start_date", "description": "month after observation date"},
            {"field_name": "perf_end_date", "description": "12-month forward performance end date"},
            {"field_name": "bad_flag", "description": "final borrower-level bad flag"},
            {"field_name": "bad_event_type", "description": "event type that triggered bad"},
            {"field_name": "bad_event_source", "description": "internal or external source"},
            {"field_name": "bad_event_date", "description": "earliest qualifying event date"},
            {"field_name": "indeterminate_flag", "description": "gray-zone flag"},
            {"field_name": "label_status", "description": "good, bad, or indeterminate"},
        ]
    )
