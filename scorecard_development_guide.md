# Credit Risk Scorecard Development — Business Walkthrough

**Context:** Car finance PD model using a `PD_cust × f(deal)` structure, with Equifax score as the customer risk proxy and deal variables (LTV, deposit %, loan term, instalment amount, etc.) as the deal component. Interaction terms are used where the multiplicative assumption fails.

---

## How to Read This Guide

Each phase covers: what you are doing and why, the specific steps and code calls, which plots to produce, what conclusions to draw, and what to do if results force a change of course. Follow the phases in order. Decision points are marked clearly — they are the moments where results determine whether you proceed, loop back, or take an alternative path.

---

## Phase 0 — Data Setup and Quality

### Purpose
Establish clean, representative development and out-of-time (OOT) samples before any modelling begins. Decisions made here propagate through every subsequent step.

### Steps

**0.1 Extract data**

```python
from pipeline import ScorecardPipeline, DataSplitConfig

pipeline = ScorecardPipeline(
    connection_string  = "mssql+pyodbc://...",
    target             = "default_flag",
    customer_variables = ["equifax_score", "annual_income",
                          "months_at_address", "employment_status"],
    deal_variables     = ["ltv_ratio", "deposit_pct",
                          "loan_term_months", "instalment_amount"],
)

split_config = DataSplitConfig.from_dates(
    dev_start = "2020-01-01", dev_end   = "2022-12-31",
    oot_start = "2023-01-01", oot_end   = "2023-12-31",
)

pipeline.extract_data(split_config)
```

**0.2 Check the basics before doing anything else**

Print and verify:
- Record counts (development ≥ 5× the number of candidate variables as a minimum)
- Bad rates (development and OOT should be broadly similar; a large divergence suggests a population shift)
- Missing rates per variable
- Date distribution (confirm no data leakage — no outcome information predating the observation point)

**Plots to produce:**
- Bad rate by origination month (line chart) — confirms the OOT period is genuinely post-development and not cherry-picked
- Missing rate heatmap by variable — flags variables that need imputation strategy before any binning

**Conclusion to draw:**
If bad rate in OOT diverges from development by more than ~3 percentage points, document the reason before proceeding. Economic shifts, product changes, or underwriting policy changes can all cause this. It does not necessarily stop development but it means your validation metrics will be conservative, and you should note this for the model governance committee.

**Change of course — if data quality fails:**
- Missing rate > 30% on a variable → treat as a separate "Missing" category in binning, not impute. Flag for discussion with the data team.
- Bad rate < 2% across the full sample → the model will have weak discrimination and calibration will be unreliable. Consider extending the observation window or broadening the definition of default.
- Fewer than 50 bads in any planned stratum → Breslow-Day and all stratified tests will be unreliable. Aggregate strata before proceeding.

---

## Phase 1 — Variable Assessment and Binning

### Purpose
Understand each variable's relationship with default, select variables for modelling, and determine how they should enter the model (WoE-binned, raw continuous, or transformed continuous).

### Steps

**1.1 WoE binning and IV calculation**

Start with equal-frequency binning for all continuous variables, then refine manually:

```python
pipeline.run_binning(
    customer_cut_points = {
        "annual_income":    [15000, 25000, 40000, 65000],
        "months_at_address": [6, 24, 60],
    },
    deal_cut_points = {
        "ltv_ratio":         [60, 80, 100, 110],
        "deposit_pct":       [5, 10, 15, 20],
        "loan_term_months":  [24, 36, 48],
        "instalment_amount": [200, 350, 500],
    },
    customer_var_types = {"employment_status": "categorical"},
)
```

**Plots to produce:**

1. **IV bar chart** — one bar per variable, coloured by rating (Useless / Weak / Medium / Strong / Suspicious). This is your first governance deliverable: it shows stakeholders which variables were assessed and which survived.

2. **WoE profile plot per variable** — x-axis = bins ordered low to high, y-axis = WoE. A clean monotonic line is ideal. Non-monotonic WoE is a signal to revise cut points.

3. **Bad rate overlay** — plot observed bad rate on the same chart as WoE (dual axis). They should tell the same directional story.

**Conclusion to draw:**
Variables with IV < 0.10 are typically excluded. Variables with IV > 0.50 require scrutiny — check for data leakage (is a post-default variable inadvertently included?). Variables in the 0.10–0.50 range are your candidates.

**Cut point refinement rules:**
- No bin should contain fewer than 5% of the population
- No bin should contain fewer than 50 bads
- WoE should be monotonic for continuous variables (exceptions exist for policy-driven variables like loan term)
- Merge bins that are too small into adjacent ones

**Change of course — if a variable fails binning:**
- Non-monotonic after manual binning → run `DealVariableLogOddsAnalysis` (Phase 1.2) to understand the shape before abandoning the variable. A U-shaped log-odds pattern is informative, not a failure.
- Low IV but strong a priori business justification → keep in the model but flag. Regulatory reviewers often expect income to be present even if its marginal IV is moderate.
- Categorical variable with too many sparse categories → group low-frequency categories into an "Other" bin based on WoE similarity.

---

**1.2 Log-odds analysis and transformation decisions**

This step determines whether deal variables need transformation before entering the model. It directly informs the `DealVariableConfig` choices in Phase 3.

```python
interaction_pipeline = InteractionScorecardPipeline(
    target         = "default_flag",
    equifax_col    = "equifax_score",
    deal_variables = ["ltv_ratio", "deposit_pct",
                      "loan_term_months", "instalment_amount"],
)

analysis = interaction_pipeline.log_odds_analysis(
    df      = dev_df,
    n_bands = 4,
    band_labels = ["Sub-Prime", "Near-Prime", "Prime", "Super-Prime"],
    run_model_comparison = True,
    plot    = True,
    plot_save_dir = "outputs/phase1_logodds",
)
```

**Plots produced (automatically, one figure per variable):**

- **Panel 1 — Overall log-odds shape:** tells you whether the variable-default relationship is linear, curved, or non-monotonic in log-odds space. This is the primary basis for your transformation decision.
- **Panel 2 — Log-odds by Equifax band:** the central diagnostic for the interaction question. Parallel lines across bands → multiplicative structure holds. Diverging lines → interaction is real.
- **Panel 3 — WoE by Equifax band:** complements Panel 2. Consistent WoE across bands → the bin is encoding a stable risk signal. Diverging WoE → the bin's meaning changes across credit quality tiers.
- **Panel 4 — Transformation comparison:** shows the original log-odds shape alongside the shape after applying the suggested transformation. A smoother, more linear post-transformation curve confirms the transformation is worthwhile.

**Summary comparison plot (model comparison figure):**
- Gini: base model vs interaction model per variable
- ΔAIC: improvement from adding the interaction term
- LR p-values: statistical significance of each interaction

**Conclusions to draw:**

| Log-odds shape   | Transformation decision | Model input mode |
|------------------|------------------------|------------------|
| Linear           | None needed            | Continuous, no transform |
| Monotonic curved | Log or sqrt            | Continuous, log/sqrt transform |
| U-shaped         | Split at inflection    | Two binary variables, or WoE with non-monotonicity accepted |
| Non-monotonic    | WoE binning            | WoE (handles shape via bins) |
| Categorical      | WoE grouping           | WoE categorical |

For the interaction question, the LR test p-value and ΔAIC from the summary plot are your evidence. If p < 0.05 and ΔAIC < −2 for a variable, the interaction term is justified statistically. If both the log-odds Panel 2 shows diverging profiles AND the LR test is significant, the evidence for interaction is strong.

**Change of course — if transformation doesn't linearise the relationship:**
A variable with a U-shaped log-odds (e.g. deposit % where very low AND very high deposits are high risk) should be split into two features: `below_threshold` and `above_threshold`. Create these as engineered variables and re-run binning from Step 1.1. Document the split point and business rationale.

---

## Phase 2 — Multiplicative Assumption Testing

### Purpose
Formally test whether the `PD = PD_cust × f(deal)` structure is valid. This determines whether you proceed with a standard scorecard, or move to the interaction model developed in Phases 3+.

### Steps

**2.1 Run Breslow-Day tests**

```python
pipeline.run_interaction_testing(
    strata_variable = "equifax_score",
    n_strata        = 3,   # Low / Medium / High credit quality
)
```

The Breslow-Day test checks: is the odds ratio between each deal variable and default consistent across Equifax score strata? If yes, the multiplicative structure holds. If no, the deal variable's effect is not uniform across credit quality — interaction terms are needed.

**Plots to produce:**

1. **Odds ratio forest plot** — one point per stratum per variable, with 95% confidence intervals. Overlapping CIs across strata → homogeneous. Non-overlapping → interaction. This is the clearest single visual for governance.

2. **P-value summary table** — Breslow-Day p-values and Tarone-corrected p-values side by side for each variable. Include the CMH result for variables that passed BD.

**Decision tree after Breslow-Day:**

```
BD p > 0.07  →  PASS  →  Run CMH test
                            CMH p < 0.05  →  Variable has independent predictive power → INCLUDE in scorecard
                            CMH p ≥ 0.05  →  Variable adds nothing beyond customer risk → EXCLUDE

BD p 0.05–0.07  →  BORDERLINE  →  Run pairwise BD to identify which stratum pair diverges
                                   Assess practical significance (are ORs meaningfully different?)
                                   Document and proceed with caution; consider interaction term

BD p < 0.05  →  FAIL  →  Run stratum diagnostics to understand root cause
                            Sparsity-driven? → Apply Tarone correction, or merge strata
                            Genuine interaction? → Proceed to Interaction Model (Phase 3)
                            Data artefact? → Investigate and re-test
```

**2.2 Run stratum diagnostics for BD failures**

```python
pipeline.run_diagnostics(
    strata_variable    = "equifax_score",
    n_strata           = 3,
    min_cell_threshold = 5,
    verbose            = True,
)
```

This produces the per-bin, per-stratum cell count table. Look at:
- `or_range`: wide range = genuine interaction (ORs are materially different across bands)
- `pct_sparse`: high sparse % = sparsity issue inflating BD (Tarone correction is appropriate)
- `zero_cells`: zero cells make BD unreliable; merge bins or strata first

**Conclusion to draw:**
If BD fails due to sparsity → apply `apply_tarone=True`. If the Tarone-corrected p-value moves above 0.05, the original failure was a sparse data artefact. Document this explicitly.

If BD fails with a wide or_range and low sparsity → the interaction is genuine. This is not a model failure; it is a finding that informs a richer model structure.

**Change of course — if all variables fail BD:**
This is not uncommon in car finance where product mix (personal contract purchase vs hire purchase) creates structural interactions. Options:
- Build separate models per product type (stratified modelling)
- Proceed to the interaction model for all deal variables
- Accept the multiplicative structure for variables where the practical OR difference is small (<0.2 on the log-odds scale), even if statistically significant

---

## Phase 3 — Model Configuration and Building

### Purpose
Build the interaction model with the correct input mode and term structure for each variable, informed by Phase 1 and Phase 2 findings.

### Steps

**3.1 Set per-variable configurations**

Using the decisions from Phases 1 and 2, configure each deal variable:

```python
from modelling.interaction_model import DealVariableConfig

deal_configs = {
    "ltv_ratio": DealVariableConfig(
        mode      = "continuous",
        transform = "log",        # log-odds analysis showed curved monotonic
        include_main        = True,
        include_interaction = True,
    ),
    "deposit_pct": DealVariableConfig(
        mode      = "continuous",
        transform = "none",       # approximately linear in log-odds space
        include_main        = True,
        include_interaction = True,
    ),
    "loan_term_months": DealVariableConfig(
        mode      = "woe",        # non-monotonic — WoE handles shape better
        include_main        = True,
        include_interaction = True,
    ),
    "instalment_amount": DealVariableConfig(
        mode      = "continuous",
        transform = "log",
        include_main        = True,
        include_interaction = True,
    ),
}
```

**3.2 Fit models across all term structures**

Run all three term structures for governance comparison:

```python
interaction_pipeline.run_binning(
    df_dev          = dev_df,
    df_oot          = oot_df,
    deal_cut_points = {"loan_term_months": [24, 36, 48]},
    deal_var_types  = {"loan_term_months": "woe"},  # WoE for non-monotonic
)

interaction_pipeline.fit_all_term_structures(
    max_combo_size = 2,  # Singles and pairs
    min_iv         = 0.10,
    max_iv         = 0.50,
    deal_configs   = deal_configs,
)
```

This fits three parallel sets of models:
- **Full:** Equifax main + deal main + interaction (the statistically correct structure)
- **Interaction only:** Equifax main + interaction (manager's comparison request — flagged with hierarchical warning)
- **Main only:** Equifax main + deal main, no interaction (baseline — quantifies what the interaction terms add)

Each model's `diagnostic_report()` outputs automatically, including the governance notice for any hierarchical violations.

**3.3 Review individual model diagnostics**

For each model, check:
- **Coefficient signs:** All main effects should have a positive relationship with risk (positive coefficient = more of this variable = higher risk). A negative coefficient on a main effect in a full model usually indicates multicollinearity.
- **Interaction coefficient sign:** Positive β_ij means the deal variable is MORE predictive for weaker-credit customers. Negative means it is MORE predictive for stronger-credit customers. Both are plausible — the sign should make business sense.
- **P-values:** Interaction terms with p > 0.05 are not statistically supported. Consider removing that specific interaction term while retaining the variable's main effect.
- **VIF:** High VIF on main effect terms (> 5) indicates multicollinearity between deal variables. High VIF on interaction terms is expected — not an error.
- **AIC and BIC:** Record for the comparison table.

**Plots to produce at this stage:**
- **Mode summary table** — confirm which variables entered as WoE vs continuous and what transform was applied. This is a governance artefact.
- **Coefficient forest plot** — visualise coefficients and 95% confidence intervals for the full model. Helps identify terms that could be dropped without material loss.

---

## Phase 4 — Model Comparison and Selection

### Purpose
Select the best model from all candidates using statistical criteria, discrimination metrics, and business judgement. Present findings in a format suitable for the model governance committee.

### Steps

**4.1 Run the comparison**

```python
interaction_pipeline.compare_models(criterion="aic")
```

This produces four comparison outputs:

1. **AIC/BIC table** — ranked by AIC. The full model will generally win on AIC (more parameters = better fit). Check ΔAIC: if the full model beats the main-only model by less than 2 AIC points, the interaction terms are adding noise, not signal. If it beats it by > 10, the interactions are genuinely important.

2. **Likelihood ratio tests** — for nested pairs. The LR test between main-only and full model directly answers: "Do the interaction terms significantly improve fit?" p < 0.05 = yes.

3. **Discrimination table** — Gini and KS for development and OOT. The full model should have higher Gini than main-only; if it does not, the interaction terms are overfitting.

4. **Coefficient comparison** — side-by-side table across all models. Look for coefficients that are stable across model structures — those are the reliable terms.

**4.2 Point contribution breakdown**

```python
interaction_pipeline.contribution_report()
```

This shows scorecard points per WoE bin at P25/P50/P75 Equifax. The key governance question is: does the interaction produce materially different scores for the same deal variable bin depending on the customer's credit quality? If yes, can you explain this to an underwriter? If the interaction produces a 20-point score difference for the same LTV between a sub-prime and a prime customer, that is both statistically and commercially meaningful, and you should be able to articulate why.

**4.3 Marginal deal effect report**

```python
interaction_pipeline.marginal_effect_report()
```

Shows the effective slope at P25/P50/P75 Equifax. A useful governance presentation: "For a prime customer, a 10% increase in LTV increases the risk score by X points. For a sub-prime customer, the same LTV increase increases the score by Y points — demonstrating the model correctly penalises higher-LTV lending more heavily for weaker credits."

**Plots to produce:**

1. **Gini comparison bar chart** — three clusters of bars (full / interaction-only / main-only), development and OOT side by side. This is the cleanest single governance chart for discrimination.

2. **AIC delta chart** — ΔAIC relative to the best model, one bar per model. Models within ΔAIC < 2 of the best are statistically indistinguishable.

3. **LR test p-value chart** — significance of each nested comparison.

4. **Score distribution plot** — histogram of scores for bads vs goods for the selected model on the development and OOT samples. Clear separation = good discrimination.

**Model selection criteria (in priority order):**

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| LR test (interaction vs main-only) | p < 0.05 | Interaction terms must be statistically justified |
| Gini drop (dev → OOT) | ≤ 0.05 | Flag if OOT Gini drops more than 5 points |
| ΔAIC vs best | < 2 | Models within this range are equivalent on fit |
| Interaction term p-values | All < 0.05 | Drop insignificant interaction terms |
| Coefficient signs | All correct direction | Negative main effect in full model = investigate |
| Business interpretability | Qualitative | Can you explain every coefficient to a credit analyst? |

**Change of course — if no model passes all criteria:**

- **Gini drop > 0.05:** The model is overfitting development. Consider: reducing `max_combo_size` to simpler models, removing interaction terms for variables with weak LR test evidence, or widening the development window.
- **All interaction terms insignificant:** Breslow-Day found interactions but the model cannot estimate them reliably. Possible causes: too few bads per stratum×bin combination; Equifax score is already capturing the interaction implicitly. Proceed with the main-effects-only model and document.
- **Coefficient sign reversal:** A deal variable's coefficient reverses sign when moving from main-only to full model. This is a multicollinearity signal — the deal variables are too correlated with each other or with Equifax. Remove the weaker variable (lower IV) and refit.
- **Interaction-only model outperforms full model on AIC:** This would be unusual. If it occurs, check for data issues — a severely imbalanced stratum or outlier Equifax values may be distorting the full model's main effect estimates.

---

## Phase 5 — Validation

### Purpose
Confirm the selected model is stable, well-calibrated, and performs comparably on unseen data before sign-off.

### Steps

**5.1 Discrimination validation**

```python
interaction_pipeline.validate(model=selected_model)
```

Key outputs:
- **Gini (Dev / OOT):** Target range for retail car finance: 0.45–0.65. Drop ≤ 0.05.
- **KS statistic:** The score value at which goods/bads are most separated. Useful for setting an initial cut-off if needed.
- **AUC:** Directly from Gini: AUC = (Gini + 1) / 2.

**5.2 Calibration validation**

Hosmer-Lemeshow test: p > 0.05 = model is well calibrated (predicted PD matches observed bad rate by score band). The observed vs predicted table should show small differences across all score bands.

If HL p < 0.05, the model's PDs are systematically off. Options:
- Recalibrate the intercept to match the observed bad rate in the development sample
- If miscalibration is concentrated in one score band, check for data quality issues in that segment

**5.3 Stability validation**

```python
# PSI: score distribution development vs OOT
# CSI: per-variable distribution development vs OOT
```

PSI thresholds:
- < 0.10 → Stable. No action.
- 0.10–0.25 → Investigate. Which variables are shifting? (Check CSI.)
- > 0.25 → Significant shift. Consider whether the model is still fit for purpose, or whether the development sample needs refreshing.

**Plots to produce:**

1. **ROC curve** — development and OOT on the same axes. Closely overlapping curves confirm the model generalises.
2. **Score distribution histogram** — bads vs goods, development vs OOT. The shape should be broadly consistent.
3. **Observed vs predicted by score band** — the calibration diagnostic. Ideally a straight diagonal line.
4. **CSI heatmap** — variables on one axis, months on the other (if monitoring data available). Highlights which variables are drifting earliest.

**Change of course — if validation fails:**

- **KS drops significantly (> 0.05) from dev to OOT:** Discrimination deteriorated. Check if the OOT period had a different product mix or a policy change. If so, note it as a known limitation; if not, consider dropping the weakest variables and refitting.
- **PSI > 0.25:** The population has shifted. The model should not go live without understanding why. Common causes in car finance: manufacturer or dealer promotional activity changing who is applying, interest rate environment changes affecting loan terms requested.
- **HL fails in OOT:** Recalibrate the intercept using the OOT sample's observed bad rate. This is a standard and acceptable action.

---

## Phase 6 — Scorecard Scaling and Output

### Purpose
Produce the final scored output and governance documentation.

### Steps

**6.1 Produce selected interaction model output**

```python
print(selected_model.coefficient_table().to_string(index=False))
contrib_df = interaction_pipeline.contribution_report(print_table=True)
```

For a selected interaction model, a single fixed points table is not valid for interaction terms because deal-variable contribution varies by Equifax score. Use the coefficient table and the P25/P50/P75 contribution table from Phase 4 as the governance output.

**6.2 Score the population**

Produce scores for the development and OOT samples. Plot the score distribution and confirm:
- The distribution is approximately bell-shaped (not heavily skewed)
- Bads score materially lower than goods
- The cut-off zone (if applicable) is sensibly positioned

**6.3 Governance documentation**

Prepare the following for the model governance committee:

| Document | Content | Phase produced |
|----------|---------|----------------|
| Variable assessment | IV table, WoE profiles, monotonicity assessment | Phase 1 |
| Transformation rationale | Log-odds shape charts, transformation comparison plots | Phase 1 |
| Multiplicative assumption evidence | BD results, forest plot, stratum diagnostics | Phase 2 |
| Model selection rationale | AIC/BIC table, LR tests, Gini comparison, term structure comparison | Phase 4 |
| Hierarchical principle note | Governance flag from interaction-only models, explanation of why full model is preferred | Phase 4 |
| Validation report | Gini, KS, HL, PSI, CSI with thresholds | Phase 5 |
| Point contribution table | Per-bin points at P25/P50/P75 Equifax for interaction variables | Phase 4 |
| Score distribution | Development and OOT histograms | Phase 6 |

---

## Summary Decision Map

```
Phase 0: Data quality
    → Bad rate < 2% or record count insufficient → STOP. Extend window.
    → Quality acceptable → CONTINUE

Phase 1: Binning & IV
    → Variable IV < 0.10 → EXCLUDE (unless business-mandated)
    → IV > 0.50 → CHECK FOR LEAKAGE
    → Log-odds non-linear → APPLY TRANSFORM or USE WoE
    → Log-odds diverges by band → INTERACTION LIKELY → Flag for Phase 2

Phase 2: Breslow-Day
    → All variables PASS → Standard scorecard (no interaction needed)
    → Variables FAIL (sparsity-driven) → Apply Tarone, retest
    → Variables FAIL (genuine) → Proceed to Interaction Model

Phase 3: Model building
    → Interaction coefficients all significant → Full model is supported
    → Some insignificant → Drop those interaction terms, keep main effects
    → Coefficient sign reversal → Investigate multicollinearity

Phase 4: Model selection
    → Full model wins on AIC + Gini + LR test → Select full model
    → Main-only model within ΔAIC < 2 → Prefer simpler model (parsimony)
    → Gini drop > 0.05 → Simplify model or extend development window

Phase 5: Validation
    → All metrics pass → Proceed to governance
    → HL fails → Recalibrate intercept
    → PSI > 0.25 → Investigate population shift before proceeding
```

---

## One-Line Code Reference

```python
# Phase 0
split_config = DataSplitConfig.from_dates(dev_start, dev_end, oot_start, oot_end)
pipeline.extract_data(split_config)

# Phase 1
pipeline.run_binning(customer_cut_points={...}, deal_cut_points={...})
interaction_pipeline.log_odds_analysis(df=dev_df, n_bands=4, plot=True,
                                        plot_save_dir="outputs/phase1")

# Phase 2
pipeline.run_interaction_testing(strata_variable="equifax_score", n_strata=3)
pipeline.run_diagnostics(strata_variable="equifax_score")

# Phase 3
interaction_pipeline.run_binning(df_dev, df_oot, deal_cut_points={...})
interaction_pipeline.fit_all_term_structures(max_combo_size=2,
                                             deal_configs=deal_configs)

# Phase 4
interaction_pipeline.compare_models(criterion="aic")
interaction_pipeline.contribution_report()
interaction_pipeline.marginal_effect_report()

# Phase 5
interaction_pipeline.validate(model=selected_model)

# Phase 6
interaction_pipeline.contribution_report()
```
