from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.bad_definition import (
    build_bad_definition_table,
    build_indeterminate_definition_table,
    build_label_structure,
)
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
)
from src.label_builder import build_label_table_schema, build_labeling_requirements
from src.modeling_scorecard import (
    build_candidate_model_cases as build_model_design_cases,
    build_coefficient_interpretation as build_model_design_interpretation,
    build_final_variable_set as build_model_design_final_variables,
    build_model_risk_notes as build_model_design_risks,
    build_modeling_workflow as build_model_design_workflow,
)
from src.pipeline_runner import run_end_to_end_design
from src.scorecard_modeling import (
    build_provisional_scorecard_table,
    build_candidate_model_cases,
    build_coefficient_interpretation,
    build_final_variable_set,
    build_model_risk_notes,
    build_modeling_workflow,
    run_proxy_scorecard_case_analysis,
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


DEFAULT_OUTPUT_PATH = Path("reports") / "retail_behavior_scorecard_development_summary.xlsx"
DEFAULT_SCORECARD_OUTPUT_PATH = Path("reports") / "retail_behavior_scorecard_points_table_provisional.xlsx"


def build_overview_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"section": "project", "item": "scorecard_type", "value": "Retail behavior scorecard"},
            {"section": "project", "item": "approach", "value": "Basel-oriented, WoE + Logistic Regression"},
            {"section": "population", "item": "unit_key", "value": "고객번호 + 기준년월"},
            {"section": "population", "item": "reference_months", "value": "2022-09, 2023-09, 2024-09"},
            {"section": "time_structure", "item": "observation_windows", "value": "1M, 3M, 6M, 12M"},
            {"section": "time_structure", "item": "performance_window", "value": "12M"},
            {"section": "split", "item": "validation1_development_validation2", "value": "2022-09 / 2023-09 / 2024-09"},
            {"section": "label", "item": "current_state", "value": "Forward 12M label unavailable; proxy snapshot label used for provisional model"},
            {"section": "model", "item": "target_final_variable_count", "value": "About 10 to 12"},
            {"section": "model", "item": "score_scaling", "value": "Anchor Score 500, PDO 40"},
        ]
    )


def export_development_workbook(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = run_end_to_end_design()

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("Overview", build_overview_sheet()),
        ("Population", pipeline["population"]),
        ("Pop_Waterfall", pipeline["population_waterfall"]),
        ("Time_Structure", build_time_structure()),
        ("Split_Summary", build_split_summary()),
        ("Bad_Definition", build_bad_definition_table()),
        ("Indeterminate", build_indeterminate_definition_table()),
        ("Label_Reqs", build_labeling_requirements()),
        ("Label_Schema", build_label_table_schema()),
        ("Base_Schema", build_base_table_schema()),
        ("Column_Groups", build_column_groups()),
        ("Raw_to_Feature", build_raw_to_feature_mapping()),
        ("DQ_Checks", build_data_quality_checks()),
        ("Candidates", build_candidate_feature_list()),
        ("Feature_Formulas", build_feature_formulas()),
        ("Feature_Dict", build_feature_dictionary()),
        ("Screening_Workflow", build_variable_screening_workflow()),
        ("Shortlist_Pool", build_shortlisted_variable_pool()),
        ("Var_Groups", build_grouped_variable_table()),
        ("Representatives", build_representative_variable_list()),
        ("Classing_Guide", build_classing_guidance()),
        ("Model_Workflow", build_modeling_workflow()),
        ("Model_Cases", build_candidate_model_cases()),
        ("Final_Variables", build_final_variable_set()),
        ("Coef_Interp", build_coefficient_interpretation()),
        ("Model_Risks", build_model_risk_notes()),
        ("Validation_Metrics", build_validation_metric_table()),
        ("Traffic_Light", build_traffic_light_framework()),
        ("Model_Design_WF", build_model_design_workflow()),
        ("Model_Design_Cases", build_model_design_cases()),
        ("Model_Design_Vars", build_model_design_final_variables()),
        ("Model_Design_Intp", build_model_design_interpretation()),
        ("Model_Design_Risk", build_model_design_risks()),
    ]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            safe_df = df.copy()
            safe_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            worksheet = writer.book[sheet_name[:31]]
            _autosize_worksheet(worksheet, safe_df)

    return output_path


def export_provisional_scorecard_workbook(
    output_path: str | Path = DEFAULT_SCORECARD_OUTPUT_PATH,
) -> Path:
    """Export a provisional scorecard points workbook using the proxy snapshot label."""
    from src.candidate_features import generate_candidate_features
    from src.label_builder import build_proxy_label_table_from_snapshot
    from src.population_definition import run_population_definition

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv("retail_credit_synthetic_dataset_v3_with_exposure_flags.csv", encoding="utf-8-sig")
    population_df, _, _ = run_population_definition()
    filtered_base = base_df.merge(
        population_df[["고객번호", "기준년월"]],
        on=["고객번호", "기준년월"],
        how="inner",
        validate="one_to_one",
    )
    feature_df = generate_candidate_features(filtered_base)
    label_df = build_proxy_label_table_from_snapshot(filtered_base)
    model_result = run_proxy_scorecard_case_analysis(feature_df=feature_df, label_df=label_df)
    points_df = build_provisional_scorecard_table(
        woe_maps=model_result["woe_maps"],
        coefficients_df=model_result["coefficients"],
    )

    overview_df = pd.DataFrame(
        [
            {"item": "status", "value": "Provisional scorecard points table"},
            {"item": "label_type", "value": "Proxy snapshot label, not forward 12M production label"},
            {"item": "best_case_id", "value": model_result["best_case_id"].iloc[0, 0]},
            {"item": "development_rows", "value": len(model_result["dev_sample"])},
            {"item": "validation_rows", "value": len(model_result["validation_sample"])},
            {"item": "oot_rows", "value": len(model_result["oot_sample"])},
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in [
            ("Overview", overview_df),
            ("Case_Comparison", model_result["case_comparison"]),
            ("Best_Features", model_result["best_features"]),
            ("Coefficients", model_result["coefficients"]),
            ("VIF", model_result["vif"]),
            ("Sign_Check", model_result["sign_check"]),
            ("Points_Table", points_df),
        ]:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            worksheet = writer.book[sheet_name[:31]]
            _autosize_worksheet(worksheet, df)

    return output_path


def _autosize_worksheet(worksheet, df: pd.DataFrame) -> None:
    for idx, column_name in enumerate(df.columns, start=1):
        max_len = max(
            [len(str(column_name))]
            + [len(str(value)) for value in df[column_name].head(300).fillna("")]
        )
        worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = min(
            max(12, max_len + 2), 60
        )
