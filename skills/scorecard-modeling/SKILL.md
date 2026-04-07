---
name: scorecard-modeling
description: Use this skill when building interpretable baseline credit models, WoE pipelines, variable screening logic, logistic regression scorecards, and rank-order evaluation in Python.
---

# Scorecard Modeling

## Purpose
Build a governable, interpretable retail credit scorecard in Python.

## Workflow
1. Confirm target and sample frame.
2. Separate development, validation, and out-of-time samples.
3. Review event rate and data sufficiency.
4. Screen variables for leakage, missingness, instability, and business plausibility.
5. Bin variables where appropriate.
6. Calculate WoE / IV if using scorecard workflow.
7. Fit logistic regression baseline.
8. Review coefficient signs and business intuition.
9. Evaluate rank ordering and stability.
10. Convert to points if required.

## Required reporting
- sample counts and event rates
- selected variables
- coefficient table
- IV / WoE summary if applicable
- KS and ROC AUC
- bad-rate monotonicity review
- score distribution
- stability commentary

## Rules
- Start with interpretable baselines.
- Reject variables that cannot be explained or governed.
- Comment on sign reversals and suspicious coefficients.
- Do not optimize only for a single metric.
- Keep binning rules reproducible.

## Python implementation guidance
Prefer:
- pandas for dataset preparation
- scikit-learn logistic regression or statsmodels for interpretable estimation
- custom, well-documented binning utilities when needed
