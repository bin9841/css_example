from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


KEY_COLUMNS: tuple[str, ...] = ("고객번호", "기준년월")

BASE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "고객번호",
    "기준년월",
    "소매카드익스포져여부",
    "활동계좌여부",
    "유효신용카드여부",
    "평점비대상상품여부",
)

SALARY_TRANSFER_COLUMNS: tuple[str, ...] = (
    "급여이체금액_M01",
    "급여이체일자_M01",
    "급여이체금액_M02",
    "급여이체일자_M02",
    "급여이체금액_M03",
    "급여이체일자_M03",
    "급여이체금액_M04",
    "급여이체일자_M04",
    "급여이체금액_M05",
    "급여이체일자_M05",
    "급여이체금액_M06",
    "급여이체일자_M06",
    "급여이체금액_M07",
    "급여이체일자_M07",
    "급여이체금액_M08",
    "급여이체일자_M08",
    "급여이체금액_M09",
    "급여이체일자_M09",
    "급여이체금액_M10",
    "급여이체일자_M10",
    "급여이체금액_M11",
    "급여이체일자_M11",
    "급여이체금액_M12",
    "급여이체일자_M12",
    "급여이체주거래여부_현재",
    "급여이체최초확인일",
)

APPLICATION_INFORMATION_COLUMNS: tuple[str, ...] = (
    "성별코드",
    "지역코드",
    "연령",
    "신청고객구분코드",
    "직업구분코드",
    "직위코드",
    "현직장입사일",
    "최초취업일",
    "사업개시일",
    "법인설립일",
    "신청소득구분코드",
    "주거형태코드",
    "사업장소유여부",
)

DEPOSIT_PERFORMANCE_COLUMNS: tuple[str, ...] = (
    "총수신평잔_M01",
    "총수신평잔_M02",
    "총수신평잔_M03",
    "총수신평잔_M04",
    "총수신평잔_M05",
    "총수신평잔_M06",
    "총수신평잔_M07",
    "총수신평잔_M08",
    "총수신평잔_M09",
    "총수신평잔_M10",
    "총수신평잔_M11",
    "총수신평잔_M12",
    "거치식평잔_M01",
    "거치식평잔_M02",
    "거치식평잔_M03",
    "거치식평잔_M04",
    "거치식평잔_M05",
    "거치식평잔_M06",
    "거치식평잔_M07",
    "거치식평잔_M08",
    "거치식평잔_M09",
    "거치식평잔_M10",
    "거치식평잔_M11",
    "거치식평잔_M12",
    "적립식평잔_M01",
    "적립식평잔_M02",
    "적립식평잔_M03",
    "적립식평잔_M04",
    "적립식평잔_M05",
    "적립식평잔_M06",
    "적립식평잔_M07",
    "적립식평잔_M08",
    "적립식평잔_M09",
    "적립식평잔_M10",
    "적립식평잔_M11",
    "적립식평잔_M12",
    "요구불평잔_M01",
    "요구불평잔_M02",
    "요구불평잔_M03",
    "요구불평잔_M04",
    "요구불평잔_M05",
    "요구불평잔_M06",
    "요구불평잔_M07",
    "요구불평잔_M08",
    "요구불평잔_M09",
    "요구불평잔_M10",
    "요구불평잔_M11",
    "요구불평잔_M12",
    "유동성수신평잔_M01",
    "유동성수신평잔_M02",
    "유동성수신평잔_M03",
    "유동성수신평잔_M04",
    "유동성수신평잔_M05",
    "유동성수신평잔_M06",
    "경험총수신평잔_M01",
    "경험총수신평잔_M02",
    "경험총수신평잔_M03",
    "경험총수신평잔_M04",
    "경험총수신평잔_M05",
    "경험총수신평잔_M06",
)

INTERNAL_DELINQUENCY_COLUMNS: tuple[str, ...] = (
    "현재연체여부",
    "최근연체시작일",
    "최근연체해제일",
    "최근연체금액",
    "최근연체사유코드",
    "연체건수_12M",
    "최대연속연체개월수_12M",
    "연체금액_M01",
    "연체금액_M02",
    "연체금액_M03",
    "연체금액_M04",
    "연체금액_M05",
    "연체금액_M06",
    "연체금액_M07",
    "연체금액_M08",
    "연체금액_M09",
    "연체금액_M10",
    "연체금액_M11",
    "연체금액_M12",
    "연체일수_M01",
    "연체일수_M02",
    "연체일수_M03",
    "연체일수_M04",
    "연체일수_M05",
    "연체일수_M06",
    "연체일수_M07",
    "연체일수_M08",
    "연체일수_M09",
    "연체일수_M10",
    "연체일수_M11",
    "연체일수_M12",
    "내부카드연체건수_12M",
    "내부여신연체건수_12M",
    "내부카드최장연체일수_12M",
    "내부여신최장연체일수_12M",
    "내부연체계좌수_12M",
    "내부미해제연체여부_현재",
)

ENTERPRISE_CREDIT_EXPOSURE_COLUMNS: tuple[str, ...] = (
    "기업신용공여기관수_3M",
    "기업신용공여기관수_6M",
    "기업신용공여기관수_12M",
    "기업신용공여총약정액_현재",
    "기업신용공여잔액_현재",
    "법인부채비율",
    "대표자보증여신보유여부",
    "기업운영자금대출보유여부",
    "기업시설자금대출보유여부",
)

EXTERNAL_LOAN_COLUMNS: tuple[str, ...] = (
    "타사대출건수_현재",
    "타사대출잔액_현재",
    "미상환캐피탈대출건수_현재",
    "미상환저축은행대출건수_현재",
    "미상환리스사대출건수_현재",
    "대부업대출보유여부_현재",
    "미상환대출기관수_현재",
    "최근3M발생미상환대출기관수",
    "최근6M발생미상환대출기관수",
    "미상환대출총건수_현재",
    "미상환대출총잔액_현재",
    "은행업권외미상환대출건수_현재",
    "담보대출제외미상환대출건수_현재",
    "신규대출발생기관수_12M",
    "제2금융업권신용공여건수_현재",
    "단위조합제외제2금융신용공여건수_현재",
)

CARD_USAGE_COLUMNS: tuple[str, ...] = (
    "총카드한도_현재",
    "총카드이용잔액_현재",
    "월상환예정금액_현재",
    "일시불이용률_M01",
    "일시불이용률_M02",
    "일시불이용률_M03",
    "일시불이용률_M04",
    "일시불이용률_M05",
    "일시불이용률_M06",
    "신판이용률_M01",
    "신판이용률_M02",
    "신판이용률_M03",
    "신판이용률_M04",
    "신판이용률_M05",
    "신판이용률_M06",
    "카드론이용률_M01",
    "카드론이용률_M02",
    "카드론이용률_M03",
    "카드론이용률_M04",
    "카드론이용률_M05",
    "카드론이용률_M06",
    "카드론이용개월수_12M",
    "현금서비스이용개월수_12M",
    "리볼빙보유개월수_12M",
    "신용카드현금서비스금액_M01",
    "신용카드현금서비스금액_M02",
    "신용카드현금서비스금액_M03",
    "신용카드현금서비스금액_M04",
    "신용카드현금서비스금액_M05",
    "신용카드현금서비스금액_M06",
    "체크포함일시불이용금액_M01",
    "체크포함일시불이용금액_M02",
    "체크포함일시불이용금액_M03",
    "체크포함일시불이용금액_M04",
    "체크포함일시불이용금액_M05",
    "체크포함일시불이용금액_M06",
    "체크카드일시불이용금액_M01",
    "체크카드일시불이용금액_M02",
    "체크카드일시불이용금액_M03",
    "체크카드일시불이용금액_M04",
    "체크카드일시불이용금액_M05",
    "체크카드일시불이용금액_M06",
    "현금서비스20만원이상이용여부_M01",
    "현금서비스20만원이상이용여부_M02",
    "현금서비스20만원이상이용여부_M03",
    "현금서비스20만원이상이용여부_M04",
    "현금서비스20만원이상이용여부_M05",
    "현금서비스20만원이상이용여부_M06",
    "신용카드사용개월수_12M",
    "체크카드사용개월수_12M",
    "미해지신용카드총건수",
    "총카드개설건수_해지포함",
)

ACCOUNT_OPENING_COLUMNS: tuple[str, ...] = (
    "거래개시일",
    "가입경과개월수",
    "CB이용개월수_12M",
    "카드보유개월수",
    "최초신용개설일",
    "최초카드개설일",
    "최초대출개설일",
    "최근신용개설일",
    "최근12M신규개설건수",
    "최근24M신규개설건수",
)

EXTERNAL_DELINQUENCY_COLUMNS: tuple[str, ...] = (
    "외부연체여부",
    "외부연체건수_12M",
    "외부연체금액_12M",
    "외부연체총건수_3M",
    "외부연체총건수_6M",
    "외부연체총건수_12M",
    "외부최장연체일수_3M",
    "외부최장연체일수_6M",
    "외부최장연체일수_12M",
    "대부업권외부연체건수_12M",
    "NICE단기연체60일이상발생여부_12M",
    "CB채무불이행등재여부_12M",
    "CB채무불이행최근발생일",
    "외부연체대부업권포함여부_12M",
)


@dataclass(frozen=True)
class ColumnGroup:
    group_name: str
    description: str
    columns: tuple[str, ...]


COLUMN_GROUPS: tuple[ColumnGroup, ...] = (
    ColumnGroup("identity_and_scope", "base key and scoring-scope controls", BASE_IDENTITY_COLUMNS),
    ColumnGroup("salary_transfer", "raw salary transfer transaction variables", SALARY_TRANSFER_COLUMNS),
    ColumnGroup("application_information", "application and borrower profile variables", APPLICATION_INFORMATION_COLUMNS),
    ColumnGroup("deposit_performance", "deposit balance and liquidity performance", DEPOSIT_PERFORMANCE_COLUMNS),
    ColumnGroup("internal_delinquency", "internal arrears and delinquency history", INTERNAL_DELINQUENCY_COLUMNS),
    ColumnGroup("enterprise_credit_exposure", "enterprise-related credit exposure variables", ENTERPRISE_CREDIT_EXPOSURE_COLUMNS),
    ColumnGroup("external_loans", "external loan and non-bank borrowing variables", EXTERNAL_LOAN_COLUMNS),
    ColumnGroup("card_usage", "card balance, usage, and utilization variables", CARD_USAGE_COLUMNS),
    ColumnGroup("account_opening_information", "relationship age and account opening history", ACCOUNT_OPENING_COLUMNS),
    ColumnGroup("external_delinquency", "external delinquency and bureau default variables", EXTERNAL_DELINQUENCY_COLUMNS),
)


RAW_TO_FEATURE_MAPPING: tuple[dict[str, str], ...] = (
    {
        "feature_family": "salary_stability",
        "raw_columns": "급여이체금액_M01~M12, 급여이체일자_M01~M12, 급여이체주거래여부_현재",
        "example_features": "salary_transfer_months_12m, avg_salary_amt_6m, salary_gap_months_12m",
        "leakage_control": "use observation-window months only; exclude any post-reference payment traces",
    },
    {
        "feature_family": "application_profile",
        "raw_columns": "연령, 직업구분코드, 직위코드, 신청소득구분코드, 주거형태코드",
        "example_features": "age_band, occupation_risk_group, housing_stability_flag",
        "leakage_control": "freeze as-of attributes at reference month; do not refresh from future applications",
    },
    {
        "feature_family": "deposit_behavior",
        "raw_columns": "총수신평잔_M01~M12, 요구불평잔_M01~M12, 유동성수신평잔_M01~M06",
        "example_features": "avg_total_deposit_3m, demand_deposit_volatility_6m, liquidity_share_1m",
        "leakage_control": "derive only from balances observed on or before reference month",
    },
    {
        "feature_family": "internal_delinquency_trend",
        "raw_columns": "현재연체여부, 연체건수_12M, 연체일수_M01~M12, 내부카드연체건수_12M",
        "example_features": "max_internal_dpd_12m, delinquent_month_count_6m, recent_internal_delinquency_flag",
        "leakage_control": "exclude performance-window arrears and target-construction future states",
    },
    {
        "feature_family": "enterprise_exposure",
        "raw_columns": "기업신용공여기관수_3M/6M/12M, 기업신용공여잔액_현재, 대표자보증여신보유여부",
        "example_features": "enterprise_exposure_flag, enterprise_balance_level, guaranteed_enterprise_loan_flag",
        "leakage_control": "treat as observation-point exposures only; no future restructuring/default outcomes",
    },
    {
        "feature_family": "external_loan_burden",
        "raw_columns": "타사대출건수_현재, 미상환대출총건수_현재, 미상환대출총잔액_현재, 신규대출발생기관수_12M",
        "example_features": "external_loan_count, nonbank_loan_balance, new_lender_count_12m",
        "leakage_control": "use snapshot or historical counts available at reference month only",
    },
    {
        "feature_family": "card_usage_intensity",
        "raw_columns": "총카드한도_현재, 총카드이용잔액_현재, 일시불이용률_M01~M06, 신판이용률_M01~M06, 카드론이용률_M01~M06",
        "example_features": "current_card_utilization, purchase_util_avg_3m, card_loan_util_trend_6m",
        "leakage_control": "derive strictly from observation-window usage history",
    },
    {
        "feature_family": "relationship_vintage",
        "raw_columns": "거래개시일, 가입경과개월수, 카드보유개월수, 최초신용개설일, 최근신용개설일",
        "example_features": "relationship_months, credit_vintage_months, recent_opening_flag_12m",
        "leakage_control": "calculate tenure relative to reference month; never use future openings",
    },
    {
        "feature_family": "external_delinquency_risk",
        "raw_columns": "외부연체건수_12M, 외부최장연체일수_12M, NICE단기연체60일이상발생여부_12M, CB채무불이행등재여부_12M",
        "example_features": "external_delinquency_count_12m, max_external_dpd_12m, cb_default_flag",
        "leakage_control": "use only delinquency history observable at reference month; do not inject future performance events",
    },
)


DATA_QUALITY_CHECKS: tuple[dict[str, str], ...] = (
    {
        "check_name": "key_uniqueness",
        "scope": "base table",
        "rule": "고객번호 + 기준년월 must be unique",
    },
    {
        "check_name": "reference_month_validity",
        "scope": "base table",
        "rule": "기준년월 must be a valid monthly snapshot and aligned to split design",
    },
    {
        "check_name": "required_raw_columns",
        "scope": "column groups",
        "rule": "all required raw columns for each group must exist before feature derivation",
    },
    {
        "check_name": "date_consistency",
        "scope": "opening and salary dates",
        "rule": "opening dates and transfer dates must not be later than the reference month end",
    },
    {
        "check_name": "balance_sign_sanity",
        "scope": "deposit, card, external loan balances",
        "rule": "numeric balances and counts should be non-negative unless business semantics say otherwise",
    },
    {
        "check_name": "flag_domain_check",
        "scope": "eligibility and delinquency flags",
        "rule": "binary flags should be restricted to expected domain values such as 0/1",
    },
    {
        "check_name": "missing_rate_summary",
        "scope": "all raw groups",
        "rule": "produce null-rate summary by column and distinguish structural missing from data defects",
    },
    {
        "check_name": "leakage_review",
        "scope": "raw-to-feature derivation",
        "rule": "exclude any post-reference-month variable or target-construction future event",
    },
)


def build_base_table_schema() -> pd.DataFrame:
    """Return the base table schema with group assignments."""
    records: list[dict[str, str]] = []
    for group in COLUMN_GROUPS:
        for column_name in group.columns:
            records.append(
                {
                    "column_name": column_name,
                    "column_group": group.group_name,
                    "description": group.description,
                    "key_flag": "Y" if column_name in KEY_COLUMNS else "N",
                }
            )
    return pd.DataFrame(records)


def build_column_groups() -> pd.DataFrame:
    """Return the column group catalog."""
    return pd.DataFrame(
        {
            "group_name": group.group_name,
            "description": group.description,
            "column_count": len(group.columns),
        }
        for group in COLUMN_GROUPS
    )


def build_raw_to_feature_mapping() -> pd.DataFrame:
    """Return the raw-to-feature mapping catalog."""
    return pd.DataFrame(RAW_TO_FEATURE_MAPPING)


def build_data_quality_checks() -> pd.DataFrame:
    """Return the data quality check list."""
    return pd.DataFrame(DATA_QUALITY_CHECKS)
