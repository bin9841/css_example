---
name: delinquency-features
description: Use this skill when creating, reviewing, or documenting delinquency-based features such as recency, frequency, severity, cure, trend, and roll-rate style indicators.
---

# Delinquency Feature Engineering

## Purpose
Create leakage-safe delinquency features for retail credit scoring.

## Typical raw inputs
- delinquency start date
- delinquency cure date
- delinquency amount
- days past due by month
- monthly delinquent balance
- count of delinquency events
- account status history

## Feature families
### 1. Recency
- days since latest delinquency start
- months since latest delinquent month
- days since latest cure

### 2. Frequency
- delinquent months in last 3/6/12 months
- delinquency events in last 12 months
- count of severe delinquency months

### 3. Severity
- max delinquency amount in last N months
- average delinquency amount in delinquent months
- max DPD in last N months
- ratio of delinquency amount to exposure

### 4. Duration
- longest consecutive delinquent streak
- total delinquent days in last N months
- active delinquency duration

### 5. Cure / persistence
- cured within 30 days flag
- unresolved delinquency flag
- repeat-delinquency-after-cure flag

### 6. Trend
- last 3 months vs prior 3 months delinquency amount change
- slope of DPD over recent months
- worsening flag based on bucket migration

## Rules
- Anchor all features to the as-of date.
- Do not use cures or delinquency events after the observation date.
- Define handling for customers with no delinquency history.
- Cap extreme values only when justified and documented.
- Keep formulas auditable.

## Output format
For each proposed feature:
1. name
2. formula
3. required raw inputs
4. window
5. missing treatment
6. interpretation
7. leakage risk
