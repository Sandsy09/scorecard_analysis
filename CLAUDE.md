# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk scorecard development toolkit for building PD (Probability of Default) models. Supports both standalone customer scorecards and interaction models combining Equifax bureau scores with deal-level variables.

## Environment Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `statsmodels`, `scikit-learn`, `sqlalchemy`, `pyodbc`, `scipy`, `pandas`, `numpy`.

No `setup.py` or `pyproject.toml` — flat package structure, import modules directly.

## Running the Code

**Interactive exploration:** `scorecard_development.ipynb` (Jupyter notebook — primary development environment)

**Standalone scorecard pipeline:**
```python
from pipeline import ScorecardPipeline
from pipeline import DataSplitConfig

config = DataSplitConfig.from_dates(
    dev_start="2020-01-01", dev_end="2021-06-30",
    val_start="2021-07-01", val_end="2021-12-31",
    oot_start="2022-01-01", oot_end="2022-12-31",
)
pipeline = ScorecardPipeline(connection_string="mssql+pyodbc://...", target="default_flag", ...)
pipeline.run_full_pipeline(split_config=config, strata_variable="credit_bureau_score", ...)
```

**Interaction model pipeline:**
```python
from interaction_pipeline import InteractionScorecardPipeline
pipeline = InteractionScorecardPipeline(target="default_flag", equifax_col="equifax_score", ...)
pipeline.fit(df_dev, df_oot, deal_cut_points={...}, max_combo_size=2)
```

**Stakeholder evidence:**
```python
from stakeholder_evidence import StakeholderEvidence
evidence = StakeholderEvidence(pipeline, equifax_col, deal_var, strata_var)
evidence.bad_rate_grid()
evidence.bd_reconciliation()
```

## Architecture

```
pipeline.py / interaction_pipeline.py   ← top-level orchestrators
    │
    ├── data/extractor.py               ← SQLAlchemy MSSQL extraction
    ├── data/sql/                       ← Raw SQL queries (combined, customer, deal, monitoring)
    │
    ├── preprocessing/binning.py        ← VariableBinner + BinningPipeline (WoE/IV)
    │
    ├── modelling/
    │   ├── logistic_model.py           ← ScorecardLogisticRegression (statsmodels-based)
    │   ├── scorecard_scaler.py         ← Log-odds → integer points (PDO formula)
    │   ├── interaction_model.py        ← InteractionLogisticRegression (Eq × deal terms)
    │   ├── model_comparison.py         ← AIC/BIC/LR tests, Gini/KS dev vs OOT
    │   ├── deal_variable_analysis.py   ← Granular interaction pattern analysis
    │   └── deal_variable_plots.py      ← Visualization of interaction effects
    │
    ├── validation/metrics.py           ← ValidationReport (Discrimination/Calibration/Stability)
    ├── testing/statistical_tests.py    ← Breslow-Day, CMH, contingency tables
    ├── reporting/stratification_table.py ← Per-stratum summaries
    │
    └── stakeholder_evidence.py         ← Evidence grids for communication
```

## Key Design Concepts

**WoE/IV binning** (`preprocessing/binning.py`): `VariableBinner` handles equal-frequency, manual cut points, and categorical binning. `BinningPipeline` orchestrates across variables. Guards enforce ≥5% population and ≥50 bads per bin. IV thresholds: useless <0.02, weak 0.02–0.10, medium 0.10–0.30, strong 0.30–0.50.

**Scorecard scaling** (`modelling/scorecard_scaler.py`): `Score = A - B × log-odds` where `B = PDO / ln(2)`. Default parameters: PDO=20, base_score=600, base_odds=50.

**Interaction model** (`modelling/interaction_model.py`): Full model is `log-odds = β₀ + β_eq·Eq + β_j·X_j + β_ij·(Eq × X_j)`. `DealVariableConfig` controls per-variable input mode (WoE vs continuous with log/sqrt transforms), which terms to include, and whether to include Equifax main effect. Hierarchical principle violations (interaction without main effects) are flagged automatically.

**Statistical testing** (`testing/statistical_tests.py`): Breslow-Day test checks homogeneity of odds ratios across Equifax strata. When BD fails (p < 0.05), CMH test determines whether a common odds ratio still exists. `InteractionTestingPipeline` orchestrates the full BD → CMH flow with `StratumDiagnostics` for triage.

**Validation** (`validation/metrics.py`): `ValidationReport` combines `DiscriminationMetrics` (Gini = 2×AUC−1, KS), `CalibrationMetrics` (Hosmer-Lemeshow), and `StabilityMetrics` (PSI, CSI). Always evaluated separately on dev, validation, and OOT.

**Model comparison** (`modelling/model_comparison.py`): `ModelComparison` ranks models by AIC/BIC, runs likelihood ratio tests, and reports point contribution breakdown at P25/P50/P75 Equifax percentiles to quantify interaction magnitude.

## Data Layer

`DataExtractor` uses SQLAlchemy context manager for MSSQL connections (`pyodbc` driver). SQL queries are in `data/sql/`. `DataSplitConfig` in `pipeline.py` supports both date-range splits and percentage-based splits.
