# AGENTS.md

## Project: Retail Behavior Scorecard (Basel-oriented)

### Core Principles
- Behavior scorecard for existing customers only
- Observation windows: 1M, 3M, 6M, 12M
- Performance period: 12 months
- Strict leakage prevention

### Development Framework
1. Population Definition (Waterfall filtering)
2. Time Structure Definition
3. Bad Definition
4. Feature Engineering
5. Modeling (Scorecard)
6. Validation (Dev / Val / OOT)

### Modeling Rules
- Use WoE + Logistic Regression
- Variable count: 10~12
- Check: IV, VIF, p-value, PSI, KS / ROC

### Validation Rules
- Stability: PSI, CAR
- Discrimination: KS, ROC, CAP
- Ranking: SDR, CDR

### Implementation
- Python-first
- Modular pipeline required
- Train / Validation / OOT strictly separated
