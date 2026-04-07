from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationMetric:
    attribute: str
    metric_name: str
    purpose: str
    traffic_light_rule: str


VALIDATION_METRICS: tuple[ValidationMetric, ...] = (
    ValidationMetric("stability", "PSI", "population and score distribution stability", "lower is better; monitor threshold crossings"),
    ValidationMetric("stability", "CAR", "characteristic stability of variable distributions", "lower deterioration is better"),
    ValidationMetric("discrimination", "CAP", "rank-ordering capture power", "higher is better"),
    ValidationMetric("discrimination", "ROC", "overall discriminatory power", "higher is better"),
    ValidationMetric("discrimination", "KS", "separation of good and bad distributions", "higher is better"),
    ValidationMetric("rank_ordering", "SDR", "score distribution ratio stability", "stable ordering required"),
    ValidationMetric("rank_ordering", "CDR", "cumulative default rate monotonicity", "stable monotonic ordering required"),
)


def build_validation_metric_table() -> pd.DataFrame:
    """Return the traffic-light validation metric framework."""
    return pd.DataFrame(asdict(item) for item in VALIDATION_METRICS)


def build_traffic_light_framework() -> pd.DataFrame:
    """Return the qualitative traffic-light action framework."""
    return pd.DataFrame(
        [
            {"traffic_light": "Green", "meaning": "performance maintained", "action": "no action required"},
            {"traffic_light": "Yellow", "meaning": "performance mildly deteriorating", "action": "monitor drivers and continue tracking"},
            {"traffic_light": "Red", "meaning": "performance materially deteriorated", "action": "recalibration, tuning, or redevelopment review"},
        ]
    )
