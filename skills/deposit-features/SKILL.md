---
name: deposit-features
description: Use this skill when engineering deposit-related features such as average balances, mix, tenure, holding months, trend, and volatility from monthly account summaries.
---

# Deposit Feature Engineering

## Purpose
Create stable deposit-behavior features relevant to repayment capacity, relationship depth, and liquidity.

## Typical raw inputs
- monthly total average deposit balance
- monthly term deposit balance
- monthly installment savings balance
- monthly demand deposit balance
- account opening date
- active account flags
- number of deposit products

## Feature families
### Balance level
- avg total deposit balance over 3/6/12 months
- latest total deposit balance
- max and min deposit balance over 12 months

### Product mix
- term deposit share
- installment savings share
- demand deposit share
- number of active deposit products

### Persistence
- months with positive term deposit balance
- months with positive installment balance
- months with any deposit relationship

### Trend
- recent 3-month average vs prior 3-month average
- month-over-month balance change
- linear trend over 6 months

### Volatility
- standard deviation of deposit balance over 6/12 months
- coefficient of variation
- count of sharp balance drops

## Rules
- Distinguish stock balances from average balances.
- Keep the window definition consistent across balance types.
- Clarify whether zero means no product, inactive product, or missing data.
- Watch for seasonality and one-off large inflows.

## Deliverables
- feature list with formulas
- expected risk direction
- missing and zero treatment
- stability concerns
