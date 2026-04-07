---
name: cb-features
description: Use this skill when working with external credit bureau variables such as score snapshots, utilization, usage ratios, months-on-book, and external delinquency indicators.
---

# External Bureau Feature Engineering

## Purpose
Design external-credit features that complement thin internal data and improve risk differentiation.

## Typical raw inputs
- bureau score
- installment utilization
- revolving utilization
- purchase usage ratio
- cash advance usage ratio
- months of credit usage
- external delinquency flags
- external delinquency count or amount
- number of open trades
- inquiry counts

## Feature families
- latest bureau score
- bureau score trend over recent months
- revolving utilization level and trend
- purchase utilization level and persistence
- months with usage in last 12 months
- external delinquency severity and recency
- trade depth and account maturity
- inquiry intensity

## Rules
- Confirm the exact bureau definition for each variable.
- Avoid mixing incomparable bureau vintages without adjustment.
- Distinguish missing bureau data from true zero activity.
- Document whether lower score means higher risk or vice versa.
- Review stability carefully if bureau field definitions vary by period.

## Output format
For each feature:
1. feature name
2. bureau source field(s)
3. transformation rule
4. interpretation
5. expected direction
6. data-quality and stability notes
