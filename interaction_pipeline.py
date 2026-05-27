"""
interaction_pipeline.py

End-to-end pipeline for building and comparing PD_overall logistic
regression models with Equifax × WoE deal variable interaction terms.

This pipeline sits alongside the existing ScorecardPipeline. It reuses
BinningPipeline for WoE transformation of deal variables, then builds
InteractionLogisticRegression models for all deal variable combinations
up to a specified size, and compares them via ModelComparison.

Context:
    The Breslow-Day test in the main ScorecardPipeline confirmed genuine
    interaction effects — the multiplicative PD_cust × f(deal) assumption
    failed. This pipeline directly addresses that by modelling the
    interaction explicitly:

        PD_overall = σ(β0
                      + β_eq  × Equifax_std
                      + β_j   × WoE_deal_j
                      + β_ij  × (Equifax_std × WoE_deal_j)
                      + ...)

    Equifax score is used as a proxy for PD_cust until the full customer
    scorecard is developed.

Steps
-----
1. run_binning()      — WoE-bin deal variables using BinningPipeline
2. fit_models()       — build all combinations up to max_combo_size,
                        fit InteractionLogisticRegression for each
3. compare_models()   — AIC/BIC, LR tests, Gini/KS, coefficient comparison
4. contribution_report() — per-bin point contributions at P25/P50/P75 Equifax

Usage
-----
    pipeline = InteractionScorecardPipeline(
        target         = "default_flag",
        equifax_col    = "equifax_score",
        deal_variables = ["ltv_ratio", "loan_term_months", "deposit_pct"],
    )

    pipeline.fit(
        df_dev          = dev_df,
        df_oot          = oot_df,
        deal_cut_points = {"ltv_ratio": [60, 80, 100]},
        max_combo_size  = 2,
    )

    # Or step-by-step:
    pipeline.run_binning(df_dev, df_oot, deal_cut_points={"ltv_ratio": [60, 80, 100]})
    pipeline.fit_models(max_combo_size=2)
    pipeline.compare_models(criterion="aic")
    contrib = pipeline.contribution_report()
"""

import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from preprocessing.binning import BinningPipeline
from modelling.interaction_model import InteractionLogisticRegression, DealVariableConfig
from modelling.model_comparison import ModelComparison
from modelling.deal_variable_analysis import DealVariableLogOddsAnalysis
from modelling.deal_variable_plots import DealVariablePlotter
from modelling.scorecard_scaler import ScorecardScaler
from validation.metrics import ValidationReport


class InteractionScorecardPipeline:
    """
    Orchestrates WoE binning, interaction model fitting, and comparison
    for the PD_overall model with Equifax × deal variable interaction terms.
    """

    def __init__(
        self,
        target:         str,
        equifax_col:    str,
        deal_variables: List[str],
        pdo:        float = 20,
        base_score: float = 600,
        base_odds:  float = 50,
    ) -> None:
        """
        Parameters
        ----------
        target         : binary target column name (1 = bad, 0 = good)
        equifax_col    : raw Equifax score column in the input dataframe
        deal_variables : list of deal variable names (raw, pre-WoE binning)
        pdo            : points to double the odds (scorecard scaling)
        base_score     : score at which base_odds applies
        base_odds      : goods:bads ratio at base_score
        """
        self.target         = target
        self.equifax_col    = equifax_col
        self.deal_variables = deal_variables
        self.pdo            = pdo
        self.base_score     = base_score
        self.base_odds      = base_odds

        # Populated during pipeline steps
        self.deal_binner:   Optional[BinningPipeline]                     = None
        self.models:        List[InteractionLogisticRegression]             = []
        self.comparison:    Optional[ModelComparison]                       = None
        self.validator:     Optional[ValidationReport]                      = None

        # Stores base deal configs (mode/transform) set by the caller;
        # term inclusion flags (include_main/interaction) are applied
        # per fit_models() call via term_structure.
        self._base_deal_configs: Optional[Dict[str, DealVariableConfig]] = None

        self.dev_data:      Optional[pd.DataFrame] = None
        self.oot_data:      Optional[pd.DataFrame] = None
        self.dev_woe:       Optional[pd.DataFrame] = None
        self.oot_woe:       Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Step 1: Binning
    # ------------------------------------------------------------------

    def run_binning(
        self,
        df_dev:          pd.DataFrame,
        df_oot:          Optional[pd.DataFrame]           = None,
        deal_cut_points: Optional[Dict[str, List[float]]] = None,
        deal_var_types:  Optional[Dict[str, str]]         = None,
        n_bins:          int                              = 10,
        min_iv:          float                            = 0.0,
        deal_configs:    Optional[Dict[str, DealVariableConfig]] = None,
    ) -> "InteractionScorecardPipeline":
        """
        Fit WoE bins for all deal variables using BinningPipeline,
        then transform development and OOT dataframes.

        Parameters
        ----------
        df_dev          : development sample
        df_oot          : out-of-time sample (optional; uses dev bins if provided)
        deal_cut_points : {variable: [cut1, cut2, ...]} for manual binning.
                          Variables not listed use equal-frequency auto-binning.
        deal_var_types  : {variable: 'categorical'} for categorical deal vars.
                          All others default to 'continuous'.
        deal_configs    : optional base DealVariableConfig by variable. The
                          interaction/main-effect flags are overwritten by
                          fit_models() term_structure, but mode/transform are
                          retained.
        n_bins          : number of equal-frequency bins for auto-binned variables
        min_iv          : minimum IV to include a deal variable in the pipeline
                          (set 0.0 to include all; raise to pre-filter weak vars)
        """
        self._log("STEP 1: Binning deal variables")

        self.dev_data = df_dev
        self.oot_data = df_oot
        self._base_deal_configs = deal_configs

        dcp = deal_cut_points or {}
        dvt = deal_var_types  or {}

        self.deal_binner = BinningPipeline(self.target)

        for var in self.deal_variables:
            self.deal_binner.add_variable(
                var,
                variable_type = dvt.get(var, "continuous"),
                cut_points    = dcp.get(var),
                n_bins        = n_bins,
            )

        self.dev_woe = self.deal_binner.fit_transform(df_dev)

        if df_oot is not None:
            self.oot_woe = self.deal_binner.transform(df_oot)

        self._log("\n  IV Summary — Deal Variables:")
        self.deal_binner.print_iv_summary()

        # Warn about any variables below min_iv
        if min_iv > 0:
            iv_summary = self.deal_binner.iv_summary
            weak_vars = iv_summary.loc[
                iv_summary["iv"] < min_iv, "variable"
            ].tolist()
            if weak_vars:
                self._log(
                    f"\n  ⚠️  Variables below IV threshold ({min_iv:.2f}) "
                    f"and excluded from model fitting: {weak_vars}"
                )

        return self

    # ------------------------------------------------------------------
    # Step 2: Build and fit models
    # ------------------------------------------------------------------

    # Valid term structure presets and what they set per variable
    _TERM_STRUCTURES = {
        "full":             {"include_main": True,  "include_interaction": True,  "include_equifax_main": True},
        "interaction_only": {"include_main": False, "include_interaction": True,  "include_equifax_main": True},
        "main_only":        {"include_main": True,  "include_interaction": False, "include_equifax_main": True},
    }

    def fit_models(
        self,
        max_combo_size:       int   = 2,
        min_iv:               float = 0.10,
        max_iv:               float = 0.50,
        term_structure:       str   = "full",
        deal_configs_override: Optional[Dict[str, "DealVariableConfig"]] = None,
    ) -> "InteractionScorecardPipeline":
        """
        Generate all deal variable combinations up to max_combo_size and
        fit an InteractionLogisticRegression for each.

        Parameters
        ----------
        max_combo_size : maximum number of deal variables per model.
                         1 → one model per deal variable.
                         2 → singles and all pairs.
                         n → all combinations up to n variables.

        min_iv / max_iv : IV filter applied before combinations are built.

        term_structure : controls which terms are included for all variables
                         in this model run.

            "full"             → Equifax main + deal main + interaction
                                 (standard, statistically sound)
            "interaction_only" → Equifax main + interaction only;
                                 deal variable main effect is dropped.
                                 ⚠️  Violates hierarchical principle.
                                 Use for governance comparison only.
            "main_only"        → Equifax main + deal main; no interaction.
                                 Useful as a baseline to quantify the
                                 value interaction terms add.

        deal_configs_override : optional per-variable DealVariableConfig dict
                                for fine-grained term control beyond what
                                term_structure provides. These override the
                                term_structure flags for named variables.
                                Variables not listed use term_structure defaults.

        Raises
        ------
        RuntimeError if run_binning() has not been called first for WoE-mode vars.
        RuntimeError if no deal variables survive the IV filter.
        ValueError if term_structure is not one of the valid values.
        """
        if term_structure not in self._TERM_STRUCTURES:
            raise ValueError(
                f"term_structure must be one of {list(self._TERM_STRUCTURES)}. "
                f"Got: '{term_structure}'."
            )

        if self.dev_woe is None:
            raise RuntimeError(
                "No WoE-transformed data found. "
                "Call run_binning() first, or ensure dev_data is set."
            )

        ts_flags = self._TERM_STRUCTURES[term_structure]

        self._log(
            f"\nSTEP 2: Fitting interaction models"
            f"\n  Term structure:   {term_structure}"
            f"\n  Max combo size:   {max_combo_size}"
            f"\n  IV filter:        [{min_iv:.2f}, {max_iv:.2f}]"
        )

        if term_structure == "interaction_only":
            self._log(
                f"  ⚠️  Note: '{term_structure}' violates the hierarchical principle. "
                "Models are flagged for comparison/governance use only."
            )

        # Variable selection — use binner IV if available, else all variables
        if self.deal_binner is not None:
            selected: List[str] = self.deal_binner.get_selected_variables(min_iv, max_iv)
            if not selected:
                raise RuntimeError(
                    f"No deal variables with IV in [{min_iv:.2f}, {max_iv:.2f}]. "
                    "Review binning or widen the IV filter."
                )
        else:
            selected = self.deal_variables

        self._log(f"  Selected variables: {selected}")

        combos: List[Tuple[str, ...]] = []
        for size in range(1, min(max_combo_size, len(selected)) + 1):
            combos.extend(itertools.combinations(selected, size))

        n_combos = len(combos)
        self._log(f"  Total models to fit: {n_combos}\n")

        for idx, combo in enumerate(combos, start=1):
            combo_list = list(combo)
            model_name = self._make_model_name(idx, combo_list, term_structure)

            self._log(f"  [{idx}/{n_combos}] {model_name}")

            # Build per-variable configs: start from term_structure defaults,
            # then apply any variable-level overrides
            resolved_configs: Dict[str, "DealVariableConfig"] = {}
            for var in combo_list:
                if deal_configs_override and var in deal_configs_override:
                    resolved_configs[var] = deal_configs_override[var]
                else:
                    # Apply term_structure flags to existing deal_configs if set
                    base_cfg = (
                        self._base_deal_configs.get(var, DealVariableConfig())
                        if hasattr(self, "_base_deal_configs") and self._base_deal_configs
                        else DealVariableConfig()
                    )
                    resolved_configs[var] = DealVariableConfig(
                        mode                = base_cfg.mode,
                        transform           = base_cfg.transform,
                        include_main        = ts_flags["include_main"],
                        include_interaction = ts_flags["include_interaction"],
                    )

            model = InteractionLogisticRegression(
                equifax_col          = self.equifax_col,
                deal_variables       = combo_list,
                target               = self.target,
                model_name           = model_name,
                deal_configs         = resolved_configs,
                include_equifax_main = ts_flags["include_equifax_main"],
            )
            model.fit(self.dev_woe)
            self.models.append(model)

            print(model.diagnostic_report())

        self._log(f"\n  All {n_combos} model(s) fitted successfully.")
        return self

    def fit_all_term_structures(
        self,
        max_combo_size: int   = 2,
        min_iv:         float = 0.10,
        max_iv:         float = 0.50,
        deal_configs:   Optional[Dict[str, DealVariableConfig]] = None,
    ) -> "InteractionScorecardPipeline":
        """
        Fit all three term structures for every variable combination and
        add all models to self.models for side-by-side comparison.

        This produces the full governance comparison set:
            - Full models       (statistically sound baseline)
            - Interaction-only  (manager's requested comparison)
            - Main-only         (confirms value of interaction terms)

        All models appear in the subsequent compare_models() output,
        grouped and labelled by term structure.

        Parameters
        ----------
        max_combo_size : maximum deal variable combination size
        min_iv / max_iv : IV filter (applied consistently across all structures)
        deal_configs : optional base DealVariableConfig by variable. Used for
                       mode/transform decisions while each term structure sets
                       include_main/include_interaction.
        """
        self._log("\nFitting all term structures for governance comparison...")
        if deal_configs is not None:
            self._base_deal_configs = deal_configs

        for structure in self._TERM_STRUCTURES:
            self._log(f"\n  --- Term structure: {structure} ---")
            self.fit_models(
                max_combo_size = max_combo_size,
                min_iv         = min_iv,
                max_iv         = max_iv,
                term_structure = structure,
            )

        self._log(
            f"\n  Total models fitted across all term structures: {len(self.models)}"
        )
        return self

    @staticmethod
    def _make_model_name(
        idx: int, deal_variables: List[str], term_structure: str = "full"
    ) -> str:
        """
        Generate a concise model name including the term structure label.
        """
        # Short suffix for term structure
        suffix_map = {
            "full":             "full",
            "interaction_only": "int_only",
            "main_only":        "main_only",
        }
        suffix = suffix_map.get(term_structure, term_structure)
        joined = "_".join(deal_variables)
        if len(joined) <= 40:
            return f"M{idx}_{joined}_{suffix}"
        preview = "_".join(deal_variables[:2])
        return f"M{idx}_{preview}_plus{len(deal_variables) - 2}more_{suffix}"

    # ------------------------------------------------------------------
    # Step 3: Compare models
    # ------------------------------------------------------------------

    def compare_models(
        self,
        criterion:  str                        = "aic",
        df_oot:     Optional[pd.DataFrame]     = None,
        y_true_oot: Optional[pd.Series]        = None,
    ) -> "InteractionScorecardPipeline":
        """
        Run the full ModelComparison suite and print results.

        Parameters
        ----------
        criterion  : 'aic' or 'bic' — used to identify and report the best model
        df_oot     : OOT dataframe for discrimination comparison (optional).
                     If not provided here, uses self.oot_woe if available.
        y_true_oot : OOT target Series. If not provided, uses self.oot_data target.

        Raises
        ------
        RuntimeError if fewer than two models have been fitted.
        """
        if not self.models:
            raise RuntimeError("Call fit_models() before compare_models().")

        if len(self.models) < 2:
            self._log(
                "\nOnly one model fitted — full comparison skipped. "
                "Increase max_combo_size or number of deal variables."
            )
            return self

        self._log("\nSTEP 3: Comparing models")

        # Resolve OOT inputs
        oot_df  = df_oot     if df_oot     is not None else self.oot_woe
        oot_y   = y_true_oot if y_true_oot is not None else (
            self.oot_data[self.target] if self.oot_data is not None else None
        )

        self.comparison = ModelComparison(self.models)
        self.comparison.print_comparison(
            df_dev     = self.dev_woe,
            y_true_dev = self.dev_data[self.target],
            df_oot     = oot_df,
            y_true_oot = oot_y,
        )

        best = self.comparison.best_model(criterion=criterion)
        self._log(
            f"  Best model by {criterion.upper()}: {best.model_name}\n"
            f"  Deal variables: {best.deal_variables}\n"
            f"  AIC={best.diagnostics['aic']:.2f}  "
            f"BIC={best.diagnostics['bic']:.2f}  "
            f"LL={best.diagnostics['log_likelihood']:.4f}"
        )

        return self

    # ------------------------------------------------------------------
    # Step 4: Point contribution report
    # ------------------------------------------------------------------

    def contribution_report(
        self,
        print_table: bool = True,
    ) -> pd.DataFrame:
        """
        Per-bin point contributions at P25/P50/P75 Equifax score for all models.

        Returns a DataFrame and optionally prints a formatted summary.

        Each row shows the scorecard points earned by a customer who:
            - is in a specific WoE bin for a deal variable
            - has an Equifax score at the 25th, 50th, or 75th percentile

        This makes the interaction tangible: the same LTV bin may be worth
        +15 points for a poor-credit customer and +6 for a prime customer.

        Raises
        ------
        RuntimeError if compare_models() has not been called first.
        RuntimeError if deal_binner is not fitted.
        """
        if self.comparison is None:
            raise RuntimeError("Call compare_models() before contribution_report().")
        if self.deal_binner is None:
            raise RuntimeError("call run_binning() before contribution_report().")

        bin_stats = self.deal_binner.get_all_bin_stats()

        contrib_df = self.comparison.point_contribution_breakdown(
            df         = self.dev_woe,
            bin_stats  = bin_stats,
            pdo        = self.pdo,
            base_score = self.base_score,
            base_odds  = self.base_odds,
        )

        if print_table:
            self._print_contribution_report(contrib_df)

        return contrib_df

    @staticmethod
    def _print_contribution_report(df: pd.DataFrame) -> None:
        sep = "=" * 72
        print(f"\n{sep}")
        print("  POINT CONTRIBUTION BREAKDOWN")
        print("  (Points per bin at P25 / P50 / P75 Equifax Score)")
        print(sep)
        print(
            "\n  Interpretation: a positive points value means this bin "
            "REDUCES the score (adds risk). Negative points mean this bin "
            "INCREASES the score (reduces risk).\n"
        )

        for model_name in df["model"].unique():
            model_df = df[df["model"] == model_name]
            print(f"\n  Model: {model_name}")
            print("  " + "-" * 60)

            for variable in model_df["variable"].unique():
                var_df = model_df[model_df["variable"] == variable]
                term_type = var_df["term_type"].iloc[0]
                print(f"\n    Variable: {variable}  ({term_type})")

                if term_type == "Equifax (Main)":
                    # Single row per percentile — no bins
                    pivot = (
                        var_df[["equifax_percentile", "points"]]
                        .set_index("equifax_percentile")
                        .T
                    )
                    print("    Points per +1 std Equifax:")
                    print("    " + pivot.to_string())

                else:
                    # Pivot: rows = bins, columns = Equifax percentiles
                    cols  = ["bin", "woe", "bad_rate", "equifax_percentile", "points"]
                    pivot = (
                        var_df[cols]
                        .pivot_table(
                            index   = ["bin", "woe", "bad_rate"],
                            columns = "equifax_percentile",
                            values  = "points",
                        )
                        .reset_index()
                    )
                    print("    " + pivot.to_string(index=False))

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------
    # Marginal deal effect report
    # ------------------------------------------------------------------

    def marginal_effect_report(
        self,
        model: Optional[InteractionLogisticRegression] = None,
    ) -> pd.DataFrame:
        """
        Show the effective slope (d log-odds / d WoE) for each deal
        variable at P25/P50/P75 Equifax, for the specified model.

        If no model is provided, uses the best model by AIC.

        This answers: "At what Equifax score level does this deal variable
        become most/least predictive?"

        Returns
        -------
        DataFrame: deal_variable | equifax_percentile | equifax_std |
                   effective_slope | direction
        """
        if model is None:
            if self.comparison is not None:
                model = self.comparison.best_model()
            elif self.models:
                model = self.models[0]
            else:
                raise RuntimeError("No models available. Call fit_models() first.")

        eq_std_vals: np.ndarray = model.scaler.transform(
            self.dev_woe[[model.equifax_col]]
        ).flatten()

        percentiles: Dict[str, float] = {
            "P25": float(np.percentile(eq_std_vals, 25)),
            "P50": float(np.percentile(eq_std_vals, 50)),
            "P75": float(np.percentile(eq_std_vals, 75)),
        }

        result = model.marginal_deal_effect(percentiles)

        print(f"\n{'=' * 60}")
        print(f"  MARGINAL DEAL EFFECT — {model.model_name}")
        print(f"  (Effective slope on log-odds per unit of WoE)")
        print(f"{'=' * 60}")
        print(result.to_string(index=False))
        print(f"{'=' * 60}\n")

        return result

    # ------------------------------------------------------------------
    # Validation for selected interaction model
    # ------------------------------------------------------------------

    def validate(
        self,
        model: Optional[InteractionLogisticRegression] = None,
    ) -> "InteractionScorecardPipeline":
        """
        Run the validation suite for a selected interaction model.

        This mirrors ScorecardPipeline.validate(), but scores the fitted
        InteractionLogisticRegression rather than the standalone customer
        scorecard model.
        """
        if model is None:
            model = self.best_model
        if model is None:
            raise RuntimeError("No model available. Call fit_models() first.")
        if self.dev_woe is None or self.dev_data is None:
            raise RuntimeError("No development data available. Call run_binning() first.")
        if self.oot_woe is None or self.oot_data is None:
            raise RuntimeError("No OOT data available for validation.")

        y_pred_dev = model.predict_proba(self.dev_woe)
        y_pred_oot = model.predict_proba(self.oot_woe)

        scaler = ScorecardScaler(self.pdo, self.base_score, self.base_odds)
        scores_dev = pd.Series(
            [scaler.pd_to_score(p) for p in y_pred_dev],
            index=self.dev_woe.index,
            name="score",
        )
        scores_oot = pd.Series(
            [scaler.pd_to_score(p) for p in y_pred_oot],
            index=self.oot_woe.index,
            name="score",
        )

        stability_cols = self._validation_input_columns(model)
        if not stability_cols:
            raise RuntimeError(
                "No model input columns available for CSI. "
                "Check model configuration and transformed data."
            )

        self.validator = ValidationReport(
            f"Interaction Model - {model.model_name}"
        )
        self.validator.run(
            y_true_dev = self.dev_data[self.target],
            y_pred_dev = pd.Series(y_pred_dev, index=self.dev_woe.index),
            scores_dev = scores_dev,
            y_true_oot = self.oot_data[self.target],
            y_pred_oot = pd.Series(y_pred_oot, index=self.oot_woe.index),
            scores_oot = scores_oot,
            vars_dev   = self.dev_woe[stability_cols],
            vars_oot   = self.oot_woe[stability_cols],
            variables  = stability_cols,
        )
        self.validator.print_report()
        return self

    def _validation_input_columns(
        self,
        model: InteractionLogisticRegression,
    ) -> List[str]:
        """Columns to use for CSI for the selected interaction model."""
        if self.dev_woe is None or self.oot_woe is None:
            return []

        candidates = [model.equifax_col]
        for var, cfg in model.deal_configs.items():
            candidates.append(f"{var}_woe" if cfg.mode == "woe" else var)

        return [
            col for col in candidates
            if col in self.dev_woe.columns and col in self.oot_woe.columns
        ]

    # ------------------------------------------------------------------
    # Log-odds analysis (exploratory — run before or after fit_models)
    # ------------------------------------------------------------------

    def log_odds_analysis(
        self,
        df:              Optional[pd.DataFrame]           = None,
        deal_cut_points: Optional[Dict[str, List[float]]] = None,
        deal_var_types:  Optional[Dict[str, str]]         = None,
        n_bins:          int                              = 10,
        n_bands:         int                              = 4,
        band_labels:     Optional[List[str]]              = None,
        run_model_comparison: bool                        = True,
        plot:            bool                             = True,
        plot_save_dir:   Optional[str]                    = None,
    ) -> Dict:
        """
        Run DealVariableLogOddsAnalysis on the development sample.

        Can be called before fit_models() as an exploratory step to
        inform binning decisions and transformation needs, or after
        to complement the model diagnostics.

        Parameters
        ----------
        df               : dataframe to analyse. Defaults to self.dev_data.
                           If run_binning() has been called, pre-computed
                           WoE columns are used in the model comparison.
        deal_cut_points  : cut points for binning within the analysis.
        deal_var_types   : variable types for the analysis.
        n_bins           : auto-bin count for log-odds analysis
        n_bands          : number of Equifax score bands (equal-frequency)
        band_labels      : labels for Equifax bands ordered low → high.
                           Defaults to Sub-Prime/Near-Prime/Prime/Super-Prime
                           for n_bands=4, or Low/Medium/High for n_bands=3.
        run_model_comparison : if True, also runs compare_interaction_models()
                               to quantify the LR test and Gini uplift per
                               deal variable independently.

        Returns
        -------
        Dict with keys:
            "log_odds_results"  : Dict[str, LogOddsResult]
            "model_comparison"  : pd.DataFrame or None
        """
        if df is None:
            df = self.dev_data if self.dev_data is not None else self.dev_woe
        if df is None:
            raise RuntimeError(
                "No dataframe available. Pass df= or call run_binning() first."
            )

        if band_labels is None:
            band_labels = (
                ["Sub-Prime", "Near-Prime", "Prime", "Super-Prime"]
                if n_bands == 4 else
                ["Low", "Medium", "High"]
                if n_bands == 3 else
                [f"Band_{i+1}" for i in range(n_bands)]
            )

        analyser = DealVariableLogOddsAnalysis(
            target      = self.target,
            equifax_col = self.equifax_col,
            n_bands     = n_bands,
            band_labels = band_labels,
        )

        self._log("\nLOG-ODDS ANALYSIS: Deal Variables vs Default")

        log_odds_results = analyser.run(
            df         = df,
            deal_vars  = self.deal_variables,
            cut_points = deal_cut_points,
            var_types  = deal_var_types,
            n_bins     = n_bins,
        )

        analyser.print_report(log_odds_results)

        model_comparison_df = None
        if run_model_comparison:
            self._log("\nWITH vs WITHOUT INTERACTION — per deal variable")

            # Reuse pre-computed WoE columns if binning has already run
            woe_available = (
                self.dev_woe is not None
                and all(f"{v}_woe" in self.dev_woe.columns
                        for v in self.deal_variables)
            )
            analysis_df = self.dev_woe if woe_available else df

            model_comparison_df = analyser.compare_interaction_models(
                df         = analysis_df,
                deal_vars  = self.deal_variables,
                cut_points = deal_cut_points if not woe_available else None,
                var_types  = deal_var_types  if not woe_available else None,
                n_bins     = n_bins,
            )

        # ----------------------------------------------------------
        # Plotting
        # ----------------------------------------------------------
        figs: List[plt.Figure] = []
        if plot:
            import matplotlib.pyplot as plt
            plotter = DealVariablePlotter(save_dir=plot_save_dir)

            # Raw df needed for transformation panel; use dev_data if available
            raw_df = self.dev_data if self.dev_data is not None else df

            figs = plotter.plot_all_variables(
                results   = log_odds_results,
                df_raw    = raw_df,
                deal_vars = self.deal_variables,
            )

            if model_comparison_df is not None:
                comp_fig = plotter.plot_model_comparison(model_comparison_df)
                figs.append(comp_fig)

        return {
            "log_odds_results": log_odds_results,
            "model_comparison": model_comparison_df,
            "figures":          figs,
        }

    def plot_prediction_analysis(
        self,
        var:           str,
        result:        Optional[Dict] = None,
        models:        Optional[List["InteractionLogisticRegression"]] = None,
        transforms:    Optional[List[str]] = None,
        band_labels:   Optional[List[str]] = None,
        n_bands:       int = 4,
        cut_points:    Optional[List[float]] = None,
        n_bins:        int = 10,
        plot_save_dir: Optional[str] = None,
    ) -> Dict[str, "plt.Figure"]:
        """
        Generate the three actual-vs-predicted log-odds charts for a single
        deal variable, using results and models already fitted in the pipeline.

        Produces:
            1. Single-model actual vs predicted  (best model by AIC)
            2. Multi-model comparison by band    (all fitted models)
            3. Transform comparison by band      (none, log, sqrt, poly2)

        Parameters
        ----------
        var           : deal variable name to analyse
        result        : optional pre-computed LogOddsResult dict from
                        log_odds_analysis(). If None and log_odds_analysis()
                        has been called previously, re-runs it silently.
        models        : models to include in the multi-model comparison.
                        Defaults to all models in self.models.
        transforms    : transforms for the transform comparison chart.
                        Defaults to ["none", "log", "sqrt", "poly2"].
        band_labels   : Equifax band labels. Defaults to 4-band labels.
        n_bands       : number of Equifax bands (used if band_labels is None)
        cut_points    : optional explicit cut points for the deal variable
        n_bins        : fallback bin count for binning
        plot_save_dir : directory to save PNGs. None = show interactively.

        Returns
        -------
        Dict with keys: "actual_vs_predicted", "model_comparison", "transform_comparison"
        """
        import matplotlib.pyplot as plt
        from scorecard.modelling.deal_variable_plots import DealVariablePlotter

        if self.dev_data is None:
            raise RuntimeError("No data available. Call run_binning() or fit() first.")

        default_band_labels = (
            band_labels or (
                ["Sub-Prime", "Near-Prime", "Prime", "Super-Prime"]
                if n_bands == 4 else
                ["Low", "Medium", "High"]
                if n_bands == 3 else
                [f"Band_{i+1}" for i in range(n_bands)]
            )
        )

        # Get LogOddsResult for this variable
        if result is None:
            self._log(f"  Running log-odds analysis for '{var}' (no result provided)...")
            lo_output = self.log_odds_analysis(
                df          = self.dev_data,
                n_bands     = len(default_band_labels),
                band_labels = default_band_labels,
                run_model_comparison = False,
                plot        = False,
            )
            result = lo_output["log_odds_results"]

        if var not in result:
            raise KeyError(
                f"'{var}' not found in log-odds results. "
                "Check the variable name or re-run log_odds_analysis()."
            )

        var_result  = result[var]
        plot_models = models or self.models
        target_col  = self.target

        plotter = DealVariablePlotter(save_dir=plot_save_dir)
        figs: Dict[str, "plt.Figure"] = {}

        # --- Chart 1: single best model actual vs predicted ---
        best = self.best_model
        if best is not None:
            self._log(f"  Chart 1: Actual vs Predicted — {best.model_name}")
            figs["actual_vs_predicted"] = plotter.plot_actual_vs_predicted(
                var         = var,
                result      = var_result,
                model       = best,
                df          = self.dev_woe if self.dev_woe is not None else self.dev_data,
                equifax_col = self.equifax_col,
                band_labels = default_band_labels,
                model_label = best.model_name,
                cut_points  = cut_points,
                n_bins      = n_bins,
            )

        # --- Chart 2: multi-model comparison ---
        if len(plot_models) >= 2:
            self._log(f"  Chart 2: Model comparison across {len(plot_models)} models")
            figs["model_comparison"] = plotter.plot_model_comparison_bands(
                var         = var,
                result      = var_result,
                models      = plot_models,
                df          = self.dev_woe if self.dev_woe is not None else self.dev_data,
                equifax_col = self.equifax_col,
                band_labels = default_band_labels,
                cut_points  = cut_points,
                n_bins      = n_bins,
            )

        # --- Chart 3: transform comparison ---
        self._log(f"  Chart 3: Transform comparison ({transforms or ['none','log','sqrt','poly2']})")
        figs["transform_comparison"] = plotter.plot_transform_comparison_bands(
            var         = var,
            result      = var_result,
            df          = self.dev_data,   # always uses raw data for transform fitting
            target_col  = target_col,
            equifax_col = self.equifax_col,
            band_labels = default_band_labels,
            transforms  = transforms,
            n_bins      = n_bins,
        )

        return figs

    # ------------------------------------------------------------------
    # Convenience: run full pipeline end-to-end
    # ------------------------------------------------------------------

    def fit(
        self,
        df_dev:          pd.DataFrame,
        df_oot:          Optional[pd.DataFrame]           = None,
        deal_cut_points: Optional[Dict[str, List[float]]] = None,
        deal_var_types:  Optional[Dict[str, str]]         = None,
        deal_configs:    Optional[Dict[str, DealVariableConfig]] = None,
        n_bins:          int                              = 10,
        max_combo_size:  int                              = 2,
        min_iv:          float                            = 0.10,
        max_iv:          float                            = 0.50,
        criterion:       str                              = "aic",
    ) -> "InteractionScorecardPipeline":
        """
        Run all pipeline steps end-to-end.

        Parameters
        ----------
        df_dev, df_oot   : development and OOT samples
        deal_cut_points  : manual cut points per variable (see run_binning)
        deal_var_types   : variable types per variable (see run_binning)
        deal_configs     : base DealVariableConfig by variable
        n_bins           : bins for auto-binned variables
        max_combo_size   : maximum deal variable combination size (see fit_models)
        min_iv / max_iv  : IV filter for variable selection (see fit_models)
        criterion        : 'aic' or 'bic' for best model selection

        Example
        -------
        pipeline.fit(
            df_dev          = dev_df,
            df_oot          = oot_df,
            deal_cut_points = {"ltv_ratio": [60, 80, 100]},
            deal_var_types  = {"employment_status": "categorical"},
            max_combo_size  = 2,
            min_iv          = 0.10,
            criterion       = "aic",
        )
        """
        return (
            self
            .run_binning(
                df_dev, df_oot, deal_cut_points, deal_var_types,
                n_bins, deal_configs=deal_configs,
            )
            .fit_models(max_combo_size, min_iv, max_iv)
            .compare_models(criterion)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_model(self) -> Optional[InteractionLogisticRegression]:
        """Best model by AIC, or None if no models fitted."""
        if self.comparison is not None:
            return self.comparison.best_model(criterion="aic")
        return self.models[0] if self.models else None

    @property
    def model_names(self) -> List[str]:
        return [m.model_name for m in self.models]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _log(message: str) -> None:
        print(message)
