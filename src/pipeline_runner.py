from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.base_dataset_schema import (
    build_base_table_schema,
    build_column_groups,
    build_data_quality_checks,
    build_raw_to_feature_mapping,
)
from src.candidate_features import (
    build_candidate_feature_list,
    build_feature_dictionary,
    build_feature_formulas,
    generate_candidate_features,
)
from src.label_builder import (
    build_label_table_from_snapshot,
    build_label_table_schema,
    build_labeling_requirements,
)
from src.population_definition import run_population_definition
from src.scorecard_modeling import (
    build_candidate_model_cases,
    build_coefficient_interpretation,
    build_final_variable_set,
    build_model_risk_notes,
    build_modeling_workflow,
)
from src.time_structure import build_split_summary, build_time_structure
from src.validation_framework import build_traffic_light_framework, build_validation_metric_table
from src.variable_screening import (
    build_classing_guidance,
    build_grouped_variable_table,
    build_representative_variable_list,
    build_shortlisted_variable_pool,
    build_variable_screening_workflow,
)


SOURCE_DATASET_NAME = "retail_credit_synthetic_dataset_v3_with_exposure_flags.csv"


def run_end_to_end_design(csv_path: str | Path = SOURCE_DATASET_NAME) -> dict[str, object]:
    """Run the currently available end-to-end design pipeline."""
    base_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    population_df, exclusion_df, waterfall_df = run_population_definition(csv_path=csv_path)
    population_keys = population_df[["고객번호", "기준년월"]].copy()
    filtered_base_df = base_df.merge(
        population_keys,
        on=["고객번호", "기준년월"],
        how="inner",
        validate="one_to_one",
    )
    feature_df = generate_candidate_features(filtered_base_df)

    results: dict[str, object] = {
        "population": population_df,
        "population_exclusions": exclusion_df,
        "population_waterfall": waterfall_df,
        "time_structure": build_time_structure(),
        "split_summary": build_split_summary(),
        "base_table_schema": build_base_table_schema(),
        "column_groups": build_column_groups(),
        "raw_to_feature_mapping": build_raw_to_feature_mapping(),
        "data_quality_checks": build_data_quality_checks(),
        "candidate_feature_list": build_candidate_feature_list(),
        "feature_formulas": build_feature_formulas(),
        "feature_dictionary": build_feature_dictionary(),
        "feature_table_preview": feature_df,
        "screening_workflow": build_variable_screening_workflow(),
        "shortlisted_variable_pool": build_shortlisted_variable_pool(),
        "grouped_variable_table": build_grouped_variable_table(),
        "representative_variable_list": build_representative_variable_list(),
        "classing_guidance": build_classing_guidance(),
        "modeling_workflow": build_modeling_workflow(),
        "candidate_model_cases": build_candidate_model_cases(),
        "final_variable_set": build_final_variable_set(),
        "coefficient_interpretation": build_coefficient_interpretation(),
        "model_risk_notes": build_model_risk_notes(),
        "validation_metric_table": build_validation_metric_table(),
        "traffic_light_framework": build_traffic_light_framework(),
        "labeling_requirements": build_labeling_requirements(),
        "label_table_schema": build_label_table_schema(),
    }

    try:
        results["label_table"] = build_label_table_from_snapshot(base_df=base_df)
    except NotImplementedError as exc:
        results["label_table"] = None
        results["labeling_blocker"] = str(exc)

    return results
