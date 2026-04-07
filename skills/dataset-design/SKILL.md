---
name: dataset-design
description: Use this skill when designing a retail credit scoring base table, observation dataset, raw schema, keys, or source-to-feature mapping in Python.
---

# Dataset Design

## Purpose
Design an execution-ready dataset structure for retail credit scoring model development.

Use this skill when the task involves:
- base-table schema design
- observation-month dataset construction
- key and grain definition
- source-table mapping
- temporal alignment and window design
- feature-ready raw column specification

## Workflow
1. Define the modeling unit.
   - Example: customer-month, account-month, application, facility, or obligor snapshot.
2. Define the primary key.
3. Define the observation date or as-of month convention.
4. Inventory required source domains.
   - internal delinquency
   - internal balances
   - deposit behavior
   - card behavior
   - external bureau fields
   - demographics or tenure
5. Map each feature family to raw fields.
6. Define rolling-window conventions explicitly.
7. Specify temporal alignment to prevent leakage.
8. Define mandatory data quality checks.

## Deliverables
- dataset grain statement
- primary key definition
- required source tables
- column-level raw schema proposal
- leakage-risk notes
- validation checklist

## Design standards
- Use precise names such as `as_of_yyyymm`, `customer_id`, `dpd_m01`, `deposit_avg_bal_m03`.
- Keep monthly history suffixes consistent.
- Avoid mixed grain in the same table unless clearly segregated.
- State whether M01 means most recent completed month or current observation month.

## Validation checklist
- key uniqueness
- one record per grain
- date consistency
- null-rate review
- duplicate-source review
- impossible-value checks
