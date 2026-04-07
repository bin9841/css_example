from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


REFERENCE_MONTHS: tuple[int, ...] = (202209, 202309, 202409)
OBSERVATION_WINDOWS: tuple[int, ...] = (1, 3, 6, 12)
PERFORMANCE_WINDOW_MONTHS = 12
OOT_REFERENCE_MONTH = 202409


@dataclass(frozen=True)
class SplitAssignment:
    reference_month: int
    split_name: str
    split_role: str


SPLIT_ASSIGNMENTS: tuple[SplitAssignment, ...] = (
    SplitAssignment(202209, "validation_1", "stability_and_discrimination_check"),
    SplitAssignment(202309, "development", "model_building"),
    SplitAssignment(202409, "validation_2_oot", "out_of_time_stability_check"),
)


def build_time_structure(reference_months: Iterable[int] = REFERENCE_MONTHS) -> pd.DataFrame:
    """Build the observation and performance window manifest."""
    split_map = {item.reference_month: item for item in SPLIT_ASSIGNMENTS}
    records: list[dict[str, object]] = []

    for reference_month in reference_months:
        as_of_period = pd.Period(str(reference_month), freq="M")
        perf_start = as_of_period + 1
        perf_end = as_of_period + PERFORMANCE_WINDOW_MONTHS
        split_info = split_map.get(reference_month)

        for observation_window_months in OBSERVATION_WINDOWS:
            obs_start = as_of_period - (observation_window_months - 1)
            records.append(
                {
                    "reference_month": reference_month,
                    "as_of_month": as_of_period.strftime("%Y-%m"),
                    "observation_window_months": observation_window_months,
                    "observation_start_month": obs_start.strftime("%Y-%m"),
                    "observation_end_month": as_of_period.strftime("%Y-%m"),
                    "performance_window_months": PERFORMANCE_WINDOW_MONTHS,
                    "performance_start_month": perf_start.strftime("%Y-%m"),
                    "performance_end_month": perf_end.strftime("%Y-%m"),
                    "split_name": split_info.split_name if split_info else "unassigned",
                    "split_role": split_info.split_role if split_info else "unassigned",
                    "oot_flag": int(reference_month == OOT_REFERENCE_MONTH),
                }
            )

    return pd.DataFrame(records).sort_values(
        ["reference_month", "observation_window_months"]
    )


def build_split_summary() -> pd.DataFrame:
    """Build the fixed train/validation/OOT split design table."""
    return pd.DataFrame(asdict(item) for item in SPLIT_ASSIGNMENTS)
