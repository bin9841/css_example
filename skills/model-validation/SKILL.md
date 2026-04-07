---
name: model-validation
description: Use this skill when validating a retail credit scoring model for discrimination, calibration, stability, documentation quality, and deployment readiness.
---

# Model Validation

## Purpose
Assess whether a retail credit model is sufficiently robust, understandable, and deployable.

## Review dimensions
1. Data integrity
2. Target integrity
3. Feature design quality
4. Discriminatory power
5. Calibration or alignment to realized bad rate
6. Stability over time
7. Segment behavior
8. Operational feasibility
9. Documentation completeness

## Typical metrics
- KS
- ROC AUC
- Gini
- PSI
- segment-level bad rate
- score-band distribution
- realized vs expected bad rate

## Required checks
- compare development vs validation vs OOT samples
- inspect score distribution shifts
- inspect event-rate shifts
- review missing-rate shifts
- review variable-level drift for key drivers
- identify weak or unstable segments

## Deliverables
- validation summary
- major issues and severity
- metric tables
- drift commentary
- remediation actions
