from __future__ import annotations

import pandas as pd


RAW_EXTERNAL_BUREAU_USAGE_MONTHS_12M = "\u0043\u0042\uc774\uc6a9\uac1c\uc6d4\uc218_12M"
RAW_EXTERNAL_BUREAU_SEVERE_DQ_60PLUS_FLAG_12M = "\u004e\u0049\u0043\u0045\ub2e8\uae30\uc5f0\uccb460\uc77c\uc774\uc0c1\ubc1c\uc0dd\uc5ec\ubd80_12M"
RAW_EXTERNAL_BUREAU_DEFAULT_FLAG_12M = "\u0043\u0042\ucc44\ubb34\ubd88\uc774\ud589\ub4f1\uc7ac\uc5ec\ubd80_12M"
RAW_EXTERNAL_BUREAU_DEFAULT_RECENT_DATE = "\u0043\u0042\ucc44\ubb34\ubd88\uc774\ud589\ucd5c\uadfc\ubc1c\uc0dd\uc77c"

ALIAS_EXTERNAL_BUREAU_USAGE_MONTHS_12M = "external_bureau_usage_months_12m"
ALIAS_EXTERNAL_BUREAU_SEVERE_DQ_60PLUS_FLAG_12M = "external_bureau_severe_dq_60plus_flag_12m"
ALIAS_EXTERNAL_BUREAU_DEFAULT_FLAG_12M = "external_bureau_default_flag_12m"
ALIAS_EXTERNAL_BUREAU_DEFAULT_RECENT_DATE = "external_bureau_default_recent_date"


PUBLIC_COLUMN_ALIASES: dict[str, str] = {
    RAW_EXTERNAL_BUREAU_USAGE_MONTHS_12M: ALIAS_EXTERNAL_BUREAU_USAGE_MONTHS_12M,
    RAW_EXTERNAL_BUREAU_SEVERE_DQ_60PLUS_FLAG_12M: ALIAS_EXTERNAL_BUREAU_SEVERE_DQ_60PLUS_FLAG_12M,
    RAW_EXTERNAL_BUREAU_DEFAULT_FLAG_12M: ALIAS_EXTERNAL_BUREAU_DEFAULT_FLAG_12M,
    RAW_EXTERNAL_BUREAU_DEFAULT_RECENT_DATE: ALIAS_EXTERNAL_BUREAU_DEFAULT_RECENT_DATE,
}


def build_public_column_alias_table() -> pd.DataFrame:
    """Return vendor-neutral aliases for raw columns that carry provider names."""
    return pd.DataFrame(
        [
            {"raw_column_name": raw_name, "public_alias_name": alias_name}
            for raw_name, alias_name in PUBLIC_COLUMN_ALIASES.items()
        ]
    )


def apply_public_column_aliases(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vendor-neutral alias columns without removing the original raw columns.

    This keeps the executable pipeline intact while allowing downstream code and
    public-facing documentation to refer to neutral names.
    """
    out = base_df.copy()
    for raw_name, alias_name in PUBLIC_COLUMN_ALIASES.items():
        if raw_name in out.columns and alias_name not in out.columns:
            out[alias_name] = out[raw_name]
    return out
