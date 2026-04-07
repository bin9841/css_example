from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    feature_group: str
    pattern_type: str
    raw_columns: str
    formula: str
    description: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "int_dq_days_max_12m",
        "internal_delinquency",
        "severity",
        "연체일수_M01~M12",
        "max(연체일수_M01..M12)",
        "12개월 내부 연체일수 최대값",
    ),
    FeatureSpec(
        "int_dq_months_freq_12m",
        "internal_delinquency",
        "frequency",
        "연체일수_M01~M12",
        "count(연체일수_Mxx > 0)",
        "12개월 내부 연체 발생 월수",
    ),
    FeatureSpec(
        "int_dq_amt_sum_6m",
        "internal_delinquency",
        "severity",
        "연체금액_M01~M06",
        "sum(연체금액_M01..M06)",
        "최근 6개월 내부 연체금액 합계",
    ),
    FeatureSpec(
        "int_dq_recency_3m_flag",
        "internal_delinquency",
        "recency",
        "연체일수_M01~M03",
        "1 if any(연체일수_M01..M03 > 0) else 0",
        "최근 3개월 내부 연체 발생 여부",
    ),
    FeatureSpec(
        "int_dq_trend_last3_vs_prev3",
        "internal_delinquency",
        "trend",
        "연체일수_M01~M06",
        "avg(연체일수_M01..M03) - avg(연체일수_M04..M06)",
        "최근 3개월 대비 직전 3개월 내부 연체일수 추세",
    ),
    FeatureSpec(
        "int_dq_volatility_6m",
        "internal_delinquency",
        "volatility",
        "연체일수_M01~M06",
        "std(연체일수_M01..M06)",
        "최근 6개월 내부 연체일수 변동성",
    ),
    FeatureSpec(
        "deposit_avg_total_bal_3m",
        "deposit_performance",
        "severity",
        "총수신평잔_M01~M03",
        "mean(총수신평잔_M01..M03)",
        "최근 3개월 총수신 평균잔액",
    ),
    FeatureSpec(
        "deposit_avg_demand_bal_6m",
        "deposit_performance",
        "severity",
        "요구불평잔_M01~M06",
        "mean(요구불평잔_M01..M06)",
        "최근 6개월 요구불 평균잔액",
    ),
    FeatureSpec(
        "deposit_trend_total_bal_3m_vs_6m",
        "deposit_performance",
        "trend",
        "총수신평잔_M01~M06",
        "mean(M01..M03) / (mean(M04..M06)+1) - 1",
        "최근 3개월과 직전 3개월 총수신 평균잔액 변화율",
    ),
    FeatureSpec(
        "deposit_volatility_total_bal_6m",
        "deposit_performance",
        "volatility",
        "총수신평잔_M01~M06",
        "std(총수신평잔_M01..M06)",
        "최근 6개월 총수신 잔액 변동성",
    ),
    FeatureSpec(
        "deposit_liquidity_share_3m",
        "deposit_performance",
        "severity",
        "유동성수신평잔_M01~M03, 경험총수신평잔_M01~M03",
        "sum(유동성수신평잔_M01..M03) / (sum(경험총수신평잔_M01..M03)+1)",
        "최근 3개월 유동성수신 비중",
    ),
    FeatureSpec(
        "salary_transfer_freq_12m",
        "salary_transfer",
        "frequency",
        "급여이체금액_M01~M12",
        "count(급여이체금액_Mxx > 0)",
        "12개월 급여이체 발생 월수",
    ),
    FeatureSpec(
        "salary_transfer_avg_amt_6m",
        "salary_transfer",
        "severity",
        "급여이체금액_M01~M06",
        "mean(nonzero 급여이체금액_M01..M06)",
        "최근 6개월 급여이체 평균금액",
    ),
    FeatureSpec(
        "salary_transfer_recency_3m_flag",
        "salary_transfer",
        "recency",
        "급여이체금액_M01~M03",
        "1 if any(급여이체금액_M01..M03 > 0) else 0",
        "최근 3개월 급여이체 발생 여부",
    ),
    FeatureSpec(
        "salary_transfer_volatility_6m",
        "salary_transfer",
        "volatility",
        "급여이체금액_M01~M06",
        "std(급여이체금액_M01..M06)",
        "최근 6개월 급여이체 금액 변동성",
    ),
    FeatureSpec(
        "salary_transfer_trend_3m_vs_6m",
        "salary_transfer",
        "trend",
        "급여이체금액_M01~M06",
        "mean(M01..M03) / (mean(M04..M06)+1) - 1",
        "최근 3개월 대비 직전 3개월 급여이체 추세",
    ),
    FeatureSpec(
        "cb_card_util_curr",
        "cb_card_usage",
        "severity",
        "총카드이용잔액_현재, 총카드한도_현재",
        "총카드이용잔액_현재 / (총카드한도_현재 + 1)",
        "관측시점 총카드 사용률",
    ),
    FeatureSpec(
        "cb_purchase_util_avg_3m",
        "cb_card_usage",
        "severity",
        "신판이용률_M01~M03",
        "mean(신판이용률_M01..M03)",
        "최근 3개월 신판 이용률 평균",
    ),
    FeatureSpec(
        "cb_cash_service_freq_6m",
        "cb_card_usage",
        "frequency",
        "현금서비스20만원이상이용여부_M01~M06",
        "sum(현금서비스20만원이상이용여부_M01..M06)",
        "최근 6개월 고액 현금서비스 이용 횟수",
    ),
    FeatureSpec(
        "cb_card_loan_util_trend_3m_vs_6m",
        "cb_card_usage",
        "trend",
        "카드론이용률_M01~M06",
        "mean(M01..M03) - mean(M04..M06)",
        "최근 3개월 대비 직전 3개월 카드론 이용률 추세",
    ),
    FeatureSpec(
        "cb_util_volatility_6m",
        "cb_card_usage",
        "volatility",
        "일시불이용률_M01~M06",
        "std(일시불이용률_M01..M06)",
        "최근 6개월 일시불 이용률 변동성",
    ),
    FeatureSpec(
        "ext_dq_max_days_12m",
        "external_delinquency",
        "severity",
        "외부최장연체일수_12M",
        "외부최장연체일수_12M",
        "12개월 외부 최장 연체일수",
    ),
    FeatureSpec(
        "ext_dq_freq_12m",
        "external_delinquency",
        "frequency",
        "외부연체총건수_12M",
        "외부연체총건수_12M",
        "12개월 외부 연체 총건수",
    ),
    FeatureSpec(
        "ext_dq_recency_3m_flag",
        "external_delinquency",
        "recency",
        "외부연체총건수_3M",
        "1 if 외부연체총건수_3M > 0 else 0",
        "최근 3개월 외부 연체 여부",
    ),
    FeatureSpec(
        "ext_cb_default_flag",
        "external_delinquency",
        "severity",
        "CB채무불이행등재여부_12M, NICE단기연체60일이상발생여부_12M",
        "max(CB채무불이행등재여부_12M, NICE단기연체60일이상발생여부_12M)",
        "외부 부도/장기연체 플래그",
    ),
    FeatureSpec(
        "loan_exposure_total_cnt",
        "loan_exposure",
        "severity",
        "타사대출건수_현재, 미상환대출총건수_현재",
        "타사대출건수_현재 + 미상환대출총건수_현재",
        "관측시점 외부 차입 총건수",
    ),
    FeatureSpec(
        "loan_exposure_total_bal",
        "loan_exposure",
        "severity",
        "타사대출잔액_현재, 미상환대출총잔액_현재",
        "타사대출잔액_현재 + 미상환대출총잔액_현재",
        "관측시점 외부 차입 총잔액",
    ),
    FeatureSpec(
        "loan_exposure_nonbank_share",
        "loan_exposure",
        "severity",
        "은행업권외미상환대출건수_현재, 미상환대출총건수_현재",
        "은행업권외미상환대출건수_현재 / (미상환대출총건수_현재 + 1)",
        "비은행권 차입 비중",
    ),
    FeatureSpec(
        "loan_new_lender_freq_12m",
        "loan_exposure",
        "frequency",
        "신규대출발생기관수_12M",
        "신규대출발생기관수_12M",
        "12개월 신규 대출기관 발생 수",
    ),
    FeatureSpec(
        "loan_recent_new_lender_trend_3m_6m",
        "loan_exposure",
        "trend",
        "최근3M발생미상환대출기관수, 최근6M발생미상환대출기관수",
        "최근3M발생미상환대출기관수 / (최근6M발생미상환대출기관수 + 1)",
        "최근 3개월 대비 6개월 신규 차입기관 집중도",
    ),
    FeatureSpec(
        "acct_tenure_months",
        "account_opening_history",
        "severity",
        "가입경과개월수",
        "가입경과개월수",
        "관측시점 전체 거래관계 경과개월수",
    ),
    FeatureSpec(
        "acct_card_tenure_months",
        "account_opening_history",
        "severity",
        "카드보유개월수",
        "카드보유개월수",
        "관측시점 카드 보유개월수",
    ),
    FeatureSpec(
        "acct_recent_opening_freq_24m",
        "account_opening_history",
        "frequency",
        "최근24M신규개설건수",
        "최근24M신규개설건수",
        "최근 24개월 신규 개설 건수",
    ),
    FeatureSpec(
        "acct_recent_opening_intensity_12m_24m",
        "account_opening_history",
        "trend",
        "최근12M신규개설건수, 최근24M신규개설건수",
        "최근12M신규개설건수 / (최근24M신규개설건수 + 1)",
        "최근 12개월 신규 개설 집중도",
    ),
)


def build_candidate_feature_list() -> pd.DataFrame:
    """Return the candidate feature catalog."""
    return pd.DataFrame(asdict(spec) for spec in FEATURE_SPECS)


def build_feature_dictionary() -> pd.DataFrame:
    """Return a compact feature dictionary."""
    return pd.DataFrame(
        {
            "feature_name": spec.feature_name,
            "feature_group": spec.feature_group,
            "pattern_type": spec.pattern_type,
            "description": spec.description,
            "raw_columns": spec.raw_columns,
        }
        for spec in FEATURE_SPECS
    )


def build_feature_formulas() -> pd.DataFrame:
    """Return feature formulas with raw-variable lineage."""
    return pd.DataFrame(
        {
            "feature_name": spec.feature_name,
            "formula": spec.formula,
            "raw_columns": spec.raw_columns,
        }
        for spec in FEATURE_SPECS
    )


def generate_candidate_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """Generate candidate features from raw variables only."""
    feature_df = base_df[["고객번호", "기준년월"]].copy()

    int_dq_days_12m = _cols(base_df, "연체일수_M", 12)
    int_dq_amt_6m = _cols(base_df, "연체금액_M", 6)
    deposit_total_3m = _cols(base_df, "총수신평잔_M", 3)
    deposit_total_6m = _cols(base_df, "총수신평잔_M", 6)
    deposit_demand_6m = _cols(base_df, "요구불평잔_M", 6)
    liquid_3m = _cols(base_df, "유동성수신평잔_M", 3)
    experienced_deposit_3m = _cols(base_df, "경험총수신평잔_M", 3)
    salary_amt_12m = _cols(base_df, "급여이체금액_M", 12)
    salary_amt_6m = _cols(base_df, "급여이체금액_M", 6)
    purchase_util_3m = _cols(base_df, "신판이용률_M", 3)
    installment_util_6m = _cols(base_df, "일시불이용률_M", 6)
    card_loan_util_6m = _cols(base_df, "카드론이용률_M", 6)
    cash_service_flag_6m = _cols(base_df, "현금서비스20만원이상이용여부_M", 6)

    feature_df["int_dq_days_max_12m"] = int_dq_days_12m.max(axis=1)
    feature_df["int_dq_months_freq_12m"] = (int_dq_days_12m > 0).sum(axis=1)
    feature_df["int_dq_amt_sum_6m"] = int_dq_amt_6m.sum(axis=1)
    feature_df["int_dq_recency_3m_flag"] = (_cols(base_df, "연체일수_M", 3) > 0).any(axis=1).astype(int)
    feature_df["int_dq_trend_last3_vs_prev3"] = _mean(_cols(base_df, "연체일수_M", 3)) - _mean(
        _slice_cols(base_df, "연체일수_M", 4, 6)
    )
    feature_df["int_dq_volatility_6m"] = _cols(base_df, "연체일수_M", 6).std(axis=1)

    feature_df["deposit_avg_total_bal_3m"] = _mean(deposit_total_3m)
    feature_df["deposit_avg_demand_bal_6m"] = _mean(deposit_demand_6m)
    feature_df["deposit_trend_total_bal_3m_vs_6m"] = _ratio(
        _mean(deposit_total_3m), _mean(_slice_cols(base_df, "총수신평잔_M", 4, 6))
    ) - 1
    feature_df["deposit_volatility_total_bal_6m"] = deposit_total_6m.std(axis=1)
    feature_df["deposit_liquidity_share_3m"] = _ratio(liquid_3m.sum(axis=1), experienced_deposit_3m.sum(axis=1))

    feature_df["salary_transfer_freq_12m"] = (salary_amt_12m > 0).sum(axis=1)
    feature_df["salary_transfer_avg_amt_6m"] = salary_amt_6m.replace(0, np.nan).mean(axis=1).fillna(0)
    feature_df["salary_transfer_recency_3m_flag"] = (_cols(base_df, "급여이체금액_M", 3) > 0).any(axis=1).astype(int)
    feature_df["salary_transfer_volatility_6m"] = salary_amt_6m.std(axis=1)
    feature_df["salary_transfer_trend_3m_vs_6m"] = _ratio(
        _mean(_cols(base_df, "급여이체금액_M", 3)),
        _mean(_slice_cols(base_df, "급여이체금액_M", 4, 6)),
    ) - 1

    feature_df["cb_card_util_curr"] = _ratio(base_df["총카드이용잔액_현재"], base_df["총카드한도_현재"])
    feature_df["cb_purchase_util_avg_3m"] = _mean(purchase_util_3m)
    feature_df["cb_cash_service_freq_6m"] = cash_service_flag_6m.sum(axis=1)
    feature_df["cb_card_loan_util_trend_3m_vs_6m"] = _mean(_cols(base_df, "카드론이용률_M", 3)) - _mean(
        _slice_cols(base_df, "카드론이용률_M", 4, 6)
    )
    feature_df["cb_util_volatility_6m"] = installment_util_6m.std(axis=1)

    feature_df["ext_dq_max_days_12m"] = base_df["외부최장연체일수_12M"]
    feature_df["ext_dq_freq_12m"] = base_df["외부연체총건수_12M"]
    feature_df["ext_dq_recency_3m_flag"] = (base_df["외부연체총건수_3M"] > 0).astype(int)
    feature_df["ext_cb_default_flag"] = base_df[
        ["CB채무불이행등재여부_12M", "NICE단기연체60일이상발생여부_12M"]
    ].max(axis=1)

    feature_df["loan_exposure_total_cnt"] = base_df["타사대출건수_현재"] + base_df["미상환대출총건수_현재"]
    feature_df["loan_exposure_total_bal"] = base_df["타사대출잔액_현재"] + base_df["미상환대출총잔액_현재"]
    feature_df["loan_exposure_nonbank_share"] = _ratio(
        base_df["은행업권외미상환대출건수_현재"], base_df["미상환대출총건수_현재"]
    )
    feature_df["loan_new_lender_freq_12m"] = base_df["신규대출발생기관수_12M"]
    feature_df["loan_recent_new_lender_trend_3m_6m"] = _ratio(
        base_df["최근3M발생미상환대출기관수"], base_df["최근6M발생미상환대출기관수"]
    )

    feature_df["acct_tenure_months"] = base_df["가입경과개월수"]
    feature_df["acct_card_tenure_months"] = base_df["카드보유개월수"]
    feature_df["acct_recent_opening_freq_24m"] = base_df["최근24M신규개설건수"]
    feature_df["acct_recent_opening_intensity_12m_24m"] = _ratio(
        base_df["최근12M신규개설건수"], base_df["최근24M신규개설건수"]
    )

    return feature_df


def _col_name(prefix: str, month_index: int) -> str:
    return f"{prefix}{month_index:02d}"


def _cols(df: pd.DataFrame, prefix: str, end_month: int) -> pd.DataFrame:
    return df[[_col_name(prefix, idx) for idx in range(1, end_month + 1)]]


def _slice_cols(df: pd.DataFrame, prefix: str, start_month: int, end_month: int) -> pd.DataFrame:
    return df[[_col_name(prefix, idx) for idx in range(start_month, end_month + 1)]]


def _mean(frame: pd.DataFrame) -> pd.Series:
    return frame.mean(axis=1)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    return numerator / safe_denominator
