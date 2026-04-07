from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REFERENCE_MONTHS: tuple[int, ...] = (202209, 202309, 202409)
SOURCE_DATASET_NAME = "retail_credit_synthetic_dataset_v3_with_exposure_flags.csv"


@dataclass(frozen=True)
class WaterfallStep:
    step_no: int
    step_name: str
    column_name: str
    keep_values: tuple[int, ...]
    exclusion_reason: str


WATERFALL_STEPS: tuple[WaterfallStep, ...] = (
    WaterfallStep(
        step_no=1,
        step_name="retail_exposure",
        column_name="소매카드익스포져여부",
        keep_values=(1,),
        exclusion_reason="non_retail_exposure",
    ),
    WaterfallStep(
        step_no=2,
        step_name="active_account",
        column_name="활동계좌여부",
        keep_values=(1,),
        exclusion_reason="inactive_account",
    ),
    WaterfallStep(
        step_no=3,
        step_name="valid_credit_card",
        column_name="유효신용카드여부",
        keep_values=(1,),
        exclusion_reason="no_valid_credit_card",
    ),
    WaterfallStep(
        step_no=4,
        step_name="scoring_product",
        column_name="평점비대상상품여부",
        keep_values=(0,),
        exclusion_reason="non_scoring_product",
    ),
    WaterfallStep(
        step_no=5,
        step_name="good_at_observation",
        column_name="현재연체여부",
        keep_values=(0,),
        exclusion_reason="bad_at_observation_point_customer",
    ),
)


def load_population_source(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw population source as UTF-8 encoded CSV."""
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def build_population_definition(
    source_df: pd.DataFrame,
    reference_months: Iterable[int] = REFERENCE_MONTHS,
    source_dataset_name: str = SOURCE_DATASET_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build development population, exclusion detail, and waterfall summary.

    Returns:
        population_df: final row-level population table with flags.
        exclusion_detail_df: excluded rows with the first exclusion reason logged.
        waterfall_log_df: step-level before/after/excluded counts by reference month.
    """
    ref_months = tuple(reference_months)
    key_cols = ["기준년월", "고객번호"]
    _validate_required_columns(source_df)

    base_df = source_df.copy()
    base_df["source_dataset_name"] = source_dataset_name
    base_df["reference_month_flag"] = base_df["기준년월"].isin(ref_months).astype(int)
    base_df["population_flag"] = 0
    base_df["card_bs_eligible_flag"] = 0
    base_df["loan_bs_eligible_flag"] = pd.NA
    base_df["exclusion_flag"] = 0
    base_df["exclusion_reason"] = pd.NA
    base_df["waterfall_step"] = pd.NA
    base_df["population_step"] = "initial_universe"

    month_filtered = base_df.loc[base_df["reference_month_flag"] == 1].copy()
    exclusion_frames = []
    waterfall_records: list[dict[str, object]] = []

    initial_months = sorted(month_filtered["기준년월"].dropna().unique().tolist())
    for month in initial_months:
        month_count = int((month_filtered["기준년월"] == month).sum())
        waterfall_records.append(
            {
                "기준년월": month,
                "step_no": 0,
                "step_name": "reference_month_filter",
                "before_cnt": month_count,
                "after_cnt": month_count,
                "excluded_cnt": 0,
                "exclusion_rate": 0.0,
                "column_name": "기준년월",
                "rule_description": f"keep in {list(ref_months)}",
            }
        )

    working_df = month_filtered.copy()
    for step in WATERFALL_STEPS:
        keep_mask = working_df[step.column_name].isin(step.keep_values)
        kept_df = working_df.loc[keep_mask].copy()
        excluded_df = working_df.loc[~keep_mask].copy()

        if not excluded_df.empty:
            excluded_df["population_flag"] = 0
            excluded_df["card_bs_eligible_flag"] = 0
            excluded_df["exclusion_flag"] = 1
            excluded_df["exclusion_reason"] = step.exclusion_reason
            excluded_df["waterfall_step"] = step.step_no
            excluded_df["population_step"] = step.step_name
            exclusion_frames.append(
                excluded_df[
                    key_cols
                    + [
                        "population_flag",
                        "reference_month_flag",
                        "card_bs_eligible_flag",
                        "loan_bs_eligible_flag",
                        "exclusion_flag",
                        "exclusion_reason",
                        "waterfall_step",
                        "population_step",
                        "source_dataset_name",
                    ]
                ]
            )

        for month, month_before in working_df.groupby("기준년월").size().items():
            month_after = int((kept_df["기준년월"] == month).sum())
            month_excluded = int(month_before - month_after)
            exclusion_rate = month_excluded / month_before if month_before else 0.0
            waterfall_records.append(
                {
                    "기준년월": int(month),
                    "step_no": step.step_no,
                    "step_name": step.step_name,
                    "before_cnt": int(month_before),
                    "after_cnt": month_after,
                    "excluded_cnt": month_excluded,
                    "exclusion_rate": exclusion_rate,
                    "column_name": step.column_name,
                    "rule_description": f"keep {step.keep_values}",
                }
            )

        working_df = kept_df

    working_df["population_flag"] = 1
    working_df["card_bs_eligible_flag"] = 1
    working_df["exclusion_flag"] = 0
    working_df["exclusion_reason"] = pd.NA
    working_df["waterfall_step"] = len(WATERFALL_STEPS)
    working_df["population_step"] = "final_population"

    population_df = working_df[
        key_cols
        + [
            "population_flag",
            "reference_month_flag",
            "card_bs_eligible_flag",
            "loan_bs_eligible_flag",
            "exclusion_flag",
            "exclusion_reason",
            "waterfall_step",
            "population_step",
            "source_dataset_name",
        ]
    ].copy()

    exclusion_detail_df = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames
        else pd.DataFrame(
            columns=
            key_cols
            + [
                "population_flag",
                "reference_month_flag",
                "card_bs_eligible_flag",
                "loan_bs_eligible_flag",
                "exclusion_flag",
                "exclusion_reason",
                "waterfall_step",
                "population_step",
                "source_dataset_name",
            ]
        )
    )
    waterfall_log_df = pd.DataFrame(waterfall_records).sort_values(
        ["기준년월", "step_no"]
    )

    _validate_population_uniqueness(population_df)
    return population_df, exclusion_detail_df, waterfall_log_df


def run_population_definition(
    csv_path: str | Path = SOURCE_DATASET_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper for the repository dataset."""
    source_df = load_population_source(csv_path)
    return build_population_definition(source_df=source_df)


def _validate_required_columns(source_df: pd.DataFrame) -> None:
    required_columns = {"기준년월", "고객번호"} | {
        step.column_name for step in WATERFALL_STEPS
    }
    missing_columns = sorted(required_columns - set(source_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _validate_population_uniqueness(population_df: pd.DataFrame) -> None:
    duplicate_mask = population_df.duplicated(subset=["기준년월", "고객번호"], keep=False)
    if duplicate_mask.any():
        duplicates = population_df.loc[duplicate_mask, ["기준년월", "고객번호"]]
        raise ValueError(
            "Population output must be unique by 기준년월 + 고객번호. "
            f"Duplicate rows found: {len(duplicates)}"
        )
