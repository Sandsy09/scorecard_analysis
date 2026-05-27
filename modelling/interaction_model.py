"""
modelling/interaction_model.py

Logistic regression with Equifax x deal variable interaction terms
for the PD_overall model structure.

Supports two input modes per deal variable and flexible term inclusion,
allowing full comparison of model structures for governance purposes.

Input modes (per variable):
    "woe"        — deal variable enters as WoE-transformed value
    "continuous" — deal variable enters as raw value, optionally
                   transformed (log / sqrt), then standardised

Term inclusion (per variable + model level):
    include_equifax_main  — whether β_eq × Equifax_std is in the model
    include_main          — whether β_j × input_j is in the model
    include_interaction   — whether β_ij × (Equifax_std × input_j) is in the model

This allows the following model structures to be compared:

    Full model (default):
        log-odds = β0 + β_eq·Eq + β_j·X_j + β_ij·(Eq × X_j)

    Interaction only (Equifax main kept, deal main dropped):
        log-odds = β0 + β_eq·Eq + β_ij·(Eq × X_j)
        Note: violates the hierarchical principle — use for
        governance comparison only, not as a deployable model.

    Main effects only (no interaction):
        log-odds = β0 + β_eq·Eq + β_j·X_j

Hierarchical principle note:
    Including an interaction term without its constituent main effects
    causes the interaction coefficient to absorb the main effect signal,
    making it biased and scale-dependent. Any model configured this way
    is flagged explicitly in the diagnostic report and coefficient table.
    These models are appropriate for governance comparison but should not
    be deployed without careful justification.

Key classes:
    DealVariableConfig             — per-variable mode, transform, and term flags
    InteractionLogisticRegression  — fits, diagnoses, and scores the model
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Optional, Tuple

_VALID_MODES      = ("woe", "continuous")
_VALID_TRANSFORMS = ("none", "log", "sqrt")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DealVariableConfig:
    """
    Per-variable configuration for InteractionLogisticRegression.

    Parameters
    ----------
    mode : "woe" | "continuous"
        How the variable enters the model.
        "woe"        -> reads '{var}_woe' column from BinningPipeline.
        "continuous" -> reads raw '{var}' column, applies transform,
                        standardises to mean=0, std=1.

    transform : "none" | "log" | "sqrt"
        Pre-processing applied in continuous mode only.
        "none" -> standardise raw values directly.
        "log"  -> log(x + shift) then standardise.
        "sqrt" -> sqrt(x + shift) then standardise.

    include_main : bool, default True
        Whether to include the main effect term β_j × input_j.
        Set False to drop the main effect while keeping the interaction.
        Note: dropping the main effect while keeping the interaction
        violates the hierarchical principle — see module docstring.

    include_interaction : bool, default True
        Whether to include the interaction term β_ij × (Equifax_std × input_j).
        Set False to fit a main-effects-only model for this variable.

    Examples
    --------
    # Full terms — standard configuration
    DealVariableConfig(mode="continuous", transform="log")

    # Interaction only — for governance comparison (hierarchical warning raised)
    DealVariableConfig(mode="continuous", transform="log",
                       include_main=False, include_interaction=True)

    # Main effect only — no interaction for this variable
    DealVariableConfig(mode="woe", include_interaction=False)
    """
    mode:                str  = "woe"
    transform:           str  = "none"
    include_main:        bool = True
    include_interaction: bool = True

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}. Got: '{self.mode}'."
            )
        if self.transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {_VALID_TRANSFORMS}. "
                f"Got: '{self.transform}'."
            )
        if self.mode == "woe" and self.transform != "none":
            raise ValueError(
                "transform is only applicable in continuous mode. "
                "Set mode='continuous' or transform='none'."
            )
        if not self.include_main and not self.include_interaction:
            raise ValueError(
                "At least one of include_main or include_interaction must be True. "
                "A variable with both set to False contributes nothing to the model."
            )


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class InteractionLogisticRegression:
    """
    Logistic regression for PD_overall with configurable term structure.

    Supports per-variable control over which terms are included, enabling
    side-by-side comparison of full, interaction-only, and main-only models.

    Design matrix columns (conditional on flags):
        equifax_std                          if include_equifax_main=True
        woe_{var}  or  cont_{var}            if include_main=True  for that var
        equifax_x_{var}                      if include_interaction=True for that var

    Backward compatible: all flags default to True, producing identical
    behaviour to the previous version when no configs are specified.
    """

    VIF_WARNING_THRESHOLD: float = 5.0
    VIF_ACTION_THRESHOLD:  float = 10.0

    def __init__(
        self,
        equifax_col:          str,
        deal_variables:       List[str],
        target:               str,
        model_name:           str = "",
        deal_configs:         Optional[Dict[str, "DealVariableConfig"]] = None,
        include_equifax_main: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        equifax_col          : raw Equifax score column (standardised internally)
        deal_variables       : deal variable names
        target               : binary target column (1=bad, 0=good)
        model_name           : label for reports and comparison tables
        deal_configs         : per-variable DealVariableConfig. Variables not
                               listed default to DealVariableConfig(mode='woe').
        include_equifax_main : whether to include β_eq × Equifax_std.
                               Set False only for experimental comparisons.
                               Note: removing the Equifax main effect when
                               interaction terms are present also violates
                               the hierarchical principle.
        """
        self.equifax_col          = equifax_col
        self.deal_variables       = deal_variables
        self.target               = target
        self.model_name           = model_name or f"Model_{'_'.join(deal_variables)}"
        self.include_equifax_main = include_equifax_main

        provided = deal_configs or {}
        self.deal_configs: Dict[str, DealVariableConfig] = {
            var: provided.get(var, DealVariableConfig(mode="woe"))
            for var in deal_variables
        }

        # WoE columns expected in the dataframe (WoE-mode vars only)
        self.woe_cols: List[str] = [
            f"{v}_woe"
            for v, cfg in self.deal_configs.items()
            if cfg.mode == "woe"
        ]

        # Set after fitting
        self.scaler:          Optional[StandardScaler]  = None
        self._cont_scalers:   Dict[str, StandardScaler] = {}
        self._cont_shifts:    Dict[str, float]          = {}
        self.sm_model                                    = None
        self.coefficients:    Optional[pd.Series]       = None
        self.intercept:       Optional[float]           = None
        self.diagnostics:     Dict                      = {}
        self._design_columns: List[str]                 = []

    # ------------------------------------------------------------------
    # Properties for introspection
    # ------------------------------------------------------------------

    @property
    def _hierarchical_violations(self) -> List[str]:
        """
        Variables where an interaction term is included but the
        corresponding main effect is dropped — a violation of the
        hierarchical principle.
        """
        violations = []
        for var, cfg in self.deal_configs.items():
            if cfg.include_interaction and not cfg.include_main:
                violations.append(var)
        if not self.include_equifax_main and any(
            cfg.include_interaction for cfg in self.deal_configs.values()
        ):
            violations.append(f"{self.equifax_col} (main effect dropped)")
        return violations

    @property
    def term_structure(self) -> str:
        """
        Human-readable label describing the overall term structure.
        Used in comparison tables and report headers.
        """
        any_main    = any(cfg.include_main for cfg in self.deal_configs.values())
        any_inter   = any(cfg.include_interaction for cfg in self.deal_configs.values())
        eq_main     = self.include_equifax_main

        if eq_main and any_main and any_inter:
            return "Full (Equifax + Deal Main + Interaction)"
        elif eq_main and not any_main and any_inter:
            return "Interaction Only (Equifax Main + Interaction, no Deal Main)"
        elif eq_main and any_main and not any_inter:
            return "Main Effects Only (no Interaction)"
        elif not eq_main and not any_main and any_inter:
            return "Interaction Term Only (no Main Effects)"
        else:
            return "Custom"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _input_col_name(self, var: str) -> str:
        """Design matrix column name for a variable's main effect term."""
        return (
            f"woe_{var}"
            if self.deal_configs[var].mode == "woe"
            else f"cont_{var}"
        )

    @staticmethod
    def _apply_transform(
        values: np.ndarray, transform: str, shift: float = 0.0
    ) -> np.ndarray:
        if transform == "log":
            return np.log(values + shift)
        elif transform == "sqrt":
            return np.sqrt(values + shift)
        return values

    def _compute_shift(self, values: np.ndarray, transform: str) -> float:
        if transform in ("log", "sqrt"):
            return max(0.0, -float(values.min())) + (1e-6 if transform == "log" else 0.0)
        return 0.0

    # ------------------------------------------------------------------
    # Design matrix
    # ------------------------------------------------------------------

    def _build_design_matrix(
        self,
        df:         pd.DataFrame,
        fit_scaler: bool = False,
    ) -> pd.DataFrame:
        """
        Build design matrix, respecting include_equifax_main,
        include_main, and include_interaction flags per variable.
        """
        # Equifax standardisation (always computed; only added if flag set)
        if fit_scaler:
            self.scaler = StandardScaler()
            equifax_std: np.ndarray = self.scaler.fit_transform(
                df[[self.equifax_col]]
            ).flatten()
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fitted. Call fit() before predict().")
            equifax_std = self.scaler.transform(
                df[[self.equifax_col]]
            ).flatten()

        X = pd.DataFrame(index=df.index)

        if self.include_equifax_main:
            X["equifax_std"] = equifax_std

        for var in self.deal_variables:
            cfg      = self.deal_configs[var]
            col_name = self._input_col_name(var)

            # --- Compute standardised input (needed for both main + interaction) ---
            if cfg.mode == "woe":
                input_vals = df[f"{var}_woe"].values.astype(float)
            else:
                raw_vals = df[var].values.astype(float)
                if fit_scaler:
                    shift       = self._compute_shift(raw_vals, cfg.transform)
                    self._cont_shifts[var] = shift
                    transformed = self._apply_transform(raw_vals, cfg.transform, shift)
                    scaler      = StandardScaler()
                    input_vals  = scaler.fit_transform(
                        transformed.reshape(-1, 1)
                    ).flatten()
                    self._cont_scalers[var] = scaler
                else:
                    if var not in self._cont_scalers:
                        raise RuntimeError(
                            f"Continuous scaler for '{var}' not fitted. "
                            "Call fit() before predict()."
                        )
                    shift       = self._cont_shifts.get(var, 0.0)
                    transformed = self._apply_transform(raw_vals, cfg.transform, shift)
                    input_vals  = self._cont_scalers[var].transform(
                        transformed.reshape(-1, 1)
                    ).flatten()

            if cfg.include_main:
                X[col_name] = input_vals

            if cfg.include_interaction:
                X[f"equifax_x_{var}"] = equifax_std * input_vals

        self._design_columns = X.columns.tolist()
        return X

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "InteractionLogisticRegression":
        """
        Fit the model. Required columns depend on configuration:
            - equifax_col        (always; raw score)
            - '{var}_woe'        for each WoE-mode variable
            - '{var}'            for each continuous-mode variable
            - target column
        """
        self._validate_inputs(df)
        X       = self._build_design_matrix(df, fit_scaler=True)
        y       = df[self.target]
        X_const = sm.add_constant(X)

        self.sm_model     = sm.Logit(y, X_const).fit(disp=0)
        self.intercept    = float(self.sm_model.params["const"])
        self.coefficients = self.sm_model.params[self._design_columns].copy()

        self._run_diagnostics(X)
        return self

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        if self.equifax_col not in df.columns:
            raise ValueError(
                f"Equifax column '{self.equifax_col}' not found in dataframe."
            )
        if self.target not in df.columns:
            raise ValueError(
                f"Target column '{self.target}' not found in dataframe."
            )
        unique_target = set(df[self.target].dropna().unique())
        if not unique_target.issubset({0, 1}):
            raise ValueError(
                f"Target must be binary (0/1). Found: {unique_target}"
            )
        missing_woe = [
            f"{v}_woe" for v, cfg in self.deal_configs.items()
            if cfg.mode == "woe" and f"{v}_woe" not in df.columns
        ]
        missing_cont = [
            v for v, cfg in self.deal_configs.items()
            if cfg.mode == "continuous" and v not in df.columns
        ]
        if missing_woe:
            raise ValueError(
                f"Missing WoE columns: {missing_woe}. "
                "Run BinningPipeline.transform() or switch to mode='continuous'."
            )
        if missing_cont:
            raise ValueError(
                f"Missing raw columns for continuous-mode variables: {missing_cont}."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _run_diagnostics(self, X: pd.DataFrame) -> None:
        vif_df = pd.DataFrame({
            "term": X.columns.tolist(),
            "vif":  [
                variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])
            ],
        })

        p_values      = self.sm_model.pvalues[self._design_columns].copy()
        insignificant = p_values[p_values > 0.05]
        high_vif      = vif_df[vif_df["vif"] > self.VIF_WARNING_THRESHOLD]

        interaction_cols       = [c for c in self._design_columns if c.startswith("equifax_x_")]
        insignificant_interact = p_values[interaction_cols][p_values[interaction_cols] > 0.05]

        self.diagnostics = {
            "vif":                        vif_df,
            "high_vif":                   high_vif,
            "p_values":                   p_values,
            "insignificant":              insignificant,
            "insignificant_interactions": insignificant_interact,
            "aic":                        self.sm_model.aic,
            "bic":                        self.sm_model.bic,
            "log_likelihood":             self.sm_model.llf,
            "n_params":                   len(self._design_columns) + 1,
            "warnings":                   self._build_warnings(
                high_vif, insignificant, insignificant_interact
            ),
        }

    def _build_warnings(
        self,
        high_vif:               pd.DataFrame,
        insignificant:          pd.Series,
        insignificant_interact: pd.Series,
    ) -> List[str]:
        warnings: List[str] = []

        # --- Hierarchical principle violations (highest priority) ---
        violations = self._hierarchical_violations
        if violations:
            warnings.append(
                f"⚠️  HIERARCHICAL PRINCIPLE VIOLATION — "
                f"interaction term(s) included without corresponding main effect(s): "
                f"{violations}. "
                "The interaction coefficient will absorb main effect signal, "
                "making it biased and scale-dependent. "
                "This model is suitable for governance comparison only. "
                "Do not deploy without explicit justification."
            )

        # --- VIF ---
        if not high_vif.empty:
            is_interact    = high_vif["term"].str.startswith("equifax_x_")
            main_high_vif  = high_vif[~is_interact]
            inter_high_vif = high_vif[is_interact]

            if not main_high_vif.empty:
                warnings.append(
                    f"⚠️  High VIF on main effect term(s): "
                    f"{main_high_vif.set_index('term')['vif'].round(2).to_dict()}. "
                    "Investigate multicollinearity between deal variables."
                )
            if not inter_high_vif.empty:
                warnings.append(
                    f"ℹ️  High VIF on interaction term(s) — expected by construction: "
                    f"{inter_high_vif.set_index('term')['vif'].round(2).to_dict()}."
                )

        # --- Insignificant terms ---
        if not insignificant_interact.empty:
            warnings.append(
                f"⚠️  Insignificant interaction term(s) (p > 0.05): "
                f"{insignificant_interact.index.tolist()}."
            )
        main_insig = insignificant[~insignificant.index.str.startswith("equifax_x_")]
        if not main_insig.empty:
            warnings.append(
                f"⚠️  Insignificant main effect(s) (p > 0.05): "
                f"{main_insig.index.tolist()}."
            )

        if not warnings:
            warnings.append("✅  All diagnostics passed.")

        return warnings

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted probability of default for each row."""
        if self.sm_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X       = self._build_design_matrix(df, fit_scaler=False)
        X_const = sm.add_constant(X, has_constant="add")
        return self.sm_model.predict(X_const).values

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def coefficient_table(self) -> pd.DataFrame:
        """
        Coefficient table with term type, input mode, p-value,
        significance, VIF, and a hierarchical_violation flag.

        Columns: term | variable | mode | type | coefficient | p_value |
                 significant | vif | hierarchical_violation
        """
        vif_series  = self.diagnostics["vif"].set_index("term")["vif"]
        violations  = set(self._hierarchical_violations)

        rows = []
        for term, coef in self.coefficients.items():
            if term.startswith("equifax_x_"):
                var       = term.replace("equifax_x_", "")
                term_type = "Interaction"
                cfg       = self.deal_configs.get(var)
                mode      = cfg.mode if cfg else "—"
                h_flag    = var in violations
            elif term == "equifax_std":
                var, term_type = self.equifax_col, "Main Effect — Equifax"
                mode = "continuous (standardised)"
                h_flag = f"{self.equifax_col} (main effect dropped)" in violations
            elif term.startswith("woe_"):
                var, term_type, mode = term.replace("woe_", ""), "Main Effect — Deal (WoE)", "woe"
                h_flag = False
            elif term.startswith("cont_"):
                var       = term.replace("cont_", "")
                cfg       = self.deal_configs.get(var)
                transform = cfg.transform if cfg else "none"
                term_type = "Main Effect — Deal (Continuous)"
                mode      = f"continuous ({transform})" if transform != "none" else "continuous"
                h_flag    = False
            else:
                var, term_type, mode, h_flag = term, "Unknown", "—", False

            rows.append({
                "term":                   term,
                "variable":               var,
                "mode":                   mode,
                "type":                   term_type,
                "coefficient":            round(float(coef), 4),
                "p_value":                round(float(self.diagnostics["p_values"][term]), 4),
                "significant":            bool(self.diagnostics["p_values"][term] < 0.05),
                "vif":                    round(float(vif_series.get(term, np.nan)), 2),
                "hierarchical_violation": h_flag,
            })

        return pd.DataFrame(rows)

    def mode_summary(self) -> pd.DataFrame:
        """
        One-row-per-variable table of configured input modes, transforms,
        and term inclusion flags.
        """
        rows = []
        for var in self.deal_variables:
            cfg = self.deal_configs[var]
            rows.append({
                "variable":          var,
                "mode":              cfg.mode,
                "transform":         cfg.transform if cfg.mode == "continuous" else "—",
                "include_main":      cfg.include_main,
                "include_interaction": cfg.include_interaction,
                "input_column":      f"{var}_woe" if cfg.mode == "woe" else var,
                "shift_applied":     (
                    round(self._cont_shifts.get(var, 0.0), 6)
                    if cfg.mode == "continuous" and self._cont_shifts
                    else "—"
                ),
            })
        return pd.DataFrame(rows)

    def marginal_deal_effect(
        self,
        equifax_percentiles: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Effective slope of each deal variable at specific Equifax levels.

            d(log-odds) / d(input_j) = β_j + β_ij × Equifax_std

        If include_main=False, β_j = 0 (main effect dropped from model).
        If include_interaction=False, β_ij = 0.

        Parameters
        ----------
        equifax_percentiles : {label: standardised Equifax value}
                              e.g. {"P25": -1.1, "P50": 0.0, "P75": 1.2}
        """
        rows = []
        for var in self.deal_variables:
            cfg      = self.deal_configs[var]
            col_name = self._input_col_name(var)

            # Coefficients — zero if the term is excluded from the model
            beta_main = (
                float(self.coefficients.get(col_name, 0.0))
                if cfg.include_main else 0.0
            )
            beta_int = (
                float(self.coefficients.get(f"equifax_x_{var}", 0.0))
                if cfg.include_interaction else 0.0
            )

            if cfg.mode == "woe":
                slope_unit = "per unit WoE"
            elif cfg.transform == "log":
                slope_unit = f"per std unit of log({var})"
            elif cfg.transform == "sqrt":
                slope_unit = f"per std unit of sqrt({var})"
            else:
                slope_unit = f"per std unit of {var}"

            for label, eq_std in equifax_percentiles.items():
                effective_slope = beta_main + beta_int * eq_std
                rows.append({
                    "deal_variable":      var,
                    "mode":               cfg.mode,
                    "include_main":       cfg.include_main,
                    "include_interaction": cfg.include_interaction,
                    "equifax_percentile": label,
                    "equifax_std":        round(eq_std, 3),
                    "effective_slope":    round(effective_slope, 4),
                    "slope_unit":         slope_unit,
                    "direction":          "Risk ↑" if effective_slope > 0 else "Risk ↓",
                })

        return pd.DataFrame(rows)

    def diagnostic_report(self) -> str:
        n_woe  = sum(1 for cfg in self.deal_configs.values() if cfg.mode == "woe")
        n_cont = sum(1 for cfg in self.deal_configs.values() if cfg.mode == "continuous")
        violations = self._hierarchical_violations

        lines = [
            "=" * 72,
            f"INTERACTION MODEL: {self.model_name}",
            f"  Term structure:  {self.term_structure}",
            f"  Deal variables:  {self.deal_variables}",
            f"  WoE-mode:        {n_woe} variable(s)",
            f"  Continuous-mode: {n_cont} variable(s)",
        ]

        if violations:
            lines += [
                "",
                "  ⚠️  GOVERNANCE NOTE — HIERARCHICAL PRINCIPLE",
                "  " + "-" * 60,
                "  This model drops main effect(s) while retaining interaction",
                "  term(s). It is configured for COMPARISON ONLY.",
                f"  Affected variables: {violations}",
                "  The interaction coefficient in this model absorbs main effect",
                "  signal and should not be interpreted in isolation.",
            ]

        lines += [
            "=" * 72,
            f"  Intercept (β0):    {self.intercept:.4f}",
            f"  AIC:               {self.diagnostics['aic']:.2f}",
            f"  BIC:               {self.diagnostics['bic']:.2f}",
            f"  Log-Likelihood:    {self.diagnostics['log_likelihood']:.4f}",
            f"  N parameters:      {self.diagnostics['n_params']}",
        ]

        if n_cont > 0:
            lines += ["", "  Input Mode Summary:"]
            for line in self.mode_summary().to_string(index=False).split("\n"):
                lines.append(f"  {line}")

        lines += [
            "",
            "  Coefficients:",
            self.coefficient_table().to_string(index=False),
            "",
            "  Warnings:",
        ]
        for w in self.diagnostics["warnings"]:
            lines.append(f"    {w}")
        lines.append("=" * 72)
        return "\n".join(lines)