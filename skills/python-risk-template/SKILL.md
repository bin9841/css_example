---
name: python-risk-template
description: Use this skill when the task requires a Python-first project structure, module layout, coding template, or reusable implementation pattern for credit scoring work.
---

# Python Risk Project Template

## Purpose
Set up a clean Python structure for retail credit scoring work.

## Recommended layout
```text
project/
├─ AGENTS.md
├─ pyproject.toml
├─ requirements.txt
├─ data/
├─ notebooks/
├─ src/
│  ├─ config.py
│  ├─ io/
│  ├─ quality/
│  ├─ features/
│  ├─ targets/
│  ├─ modeling/
│  ├─ validation/
│  └─ utils/
├─ reports/
├─ artifacts/
└─ tests/
```

## Implementation conventions
- put reusable business logic in `src/`
- keep notebooks for exploration and review only
- isolate feature logic by domain
- store schema expectations in code or YAML
- write lightweight tests for critical joins and feature calculations

## Minimum modules
- dataset builder
- target labeler
- delinquency feature generator
- deposit feature generator
- bureau feature generator
- train/validation/oot splitter
- evaluation metrics module
- documentation helper

## Output standards
- versioned artifacts
- reproducible config
- deterministic random seeds where relevant
- explicit date parameters
