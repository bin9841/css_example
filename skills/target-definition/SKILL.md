---
name: target-definition
description: Use this skill when defining bad criteria, observation windows, performance windows, exclusions, and sample labels for retail credit scoring.
---

# Target Definition

## Purpose
Define a precise, auditable target variable for retail credit scoring.

## Required decisions
1. Observation date convention
2. Performance window length
3. Bad definition
4. Good definition
5. Indeterminate / reject / exclusion handling
6. Cure rule if used
7. Multiple-account aggregation rule
8. Label assignment precedence

## Common bad definitions
- 30+ DPD within 12 months
- 60+ DPD within 12 months
- charge-off / default / write-off
- restructuring or legal collection event
- externally observed serious delinquency

## Checks
- Is the label aligned to business use?
- Is the target observable for all included records?
- Are censoring rules explicit?
- Are thin-history customers handled consistently?
- Is there any leakage from post-observation information entering features?

## Deliverables
- formal label definition
- sample inclusion and exclusion rules
- implementation pseudocode
- edge-case handling notes
- validation checks for label consistency
