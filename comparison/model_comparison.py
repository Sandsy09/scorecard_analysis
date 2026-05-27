"""
scorecard/comparison/model_comparison.py

Multi-model comparison framework for the PD_cust x f(deal) scorecard.

This module sits on top of the existing scorecard package and is designed
to evaluate arbitrary candidate model specifications side-by-side. It
fits each model, captures discrimination / calibration / stability
metrics on development and OOT samples, and exports:

    1. A long-format Excel file optimised for Power BI ingestion
    2. A self-contained interactive HTML prototype (Plotly)

Each model is defined by a ModelSpec, which lists the deal variables to
include, how each one should enter the model (WoE bin values, or raw
continuous values with an optional log / sqrt transform), and which of
them should have an interaction with the Equifax score.

Typical usage
-------------
    from scorecard.comparison import (
        VariableConfig, ModelSpec, ModelComparison,
    )

    specs = [
        ModelSpec(
            name           = "M1_DepositInt",
            deal_variables = ["deposit_value", "instalment"],
            configs        = {
                "deposit_value": VariableConfig(mode="continuous", transform="none"),
                "instalment":    VariableConfig(mode="continuous", transform="none"),
            },
            interactions   = ["deposit_value"],
        ),
        # ... further specs ...
    ]

    comparison = ModelComparison(
        specs       = specs,
        target      = "default_flag",
        equifax_col = "equifax_score",
    )
    comparison.fit(dev_df)
    comparison.evaluate(dev_df, oot_df)
    comparison.export_excel("outputs/model_comparison.xlsx")
    comparison.export_html("outputs/model_comparison.html")
    print(comparison.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Reuse the existing validation primitives so metric definitions stay
# consistent with the rest of the pipeline
from validation.metrics import (
    DiscriminationMetrics,
    CalibrationMetrics,
    StabilityMetrics,
)


_VALID_MODES      = ("woe", "continuous")
_VALID_TRANSFORMS = ("none", "log", "sqrt")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VariableConfig:
    """
    Per-variable input configuration.

    Parameters
    ----------
    mode : "woe" | "continuous"
        "woe"        → reads the '{var}_woe' column produced by BinningPipeline.
        "continuous" → reads the raw '{var}' column, applies `transform`,
                       then standardises to mean=0, std=1.

    transform : "none" | "log" | "sqrt"
        Only applied when mode="continuous". Ignored in WoE mode.
    """
    mode: str = "woe"
    transform: str = "none"

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got '{self.mode}'"
            )
        if self.transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {_VALID_TRANSFORMS}, "
                f"got '{self.transform}'"
            )
        if self.mode == "woe" and self.transform != "none":
            warnings.warn(
                f"transform='{self.transform}' is ignored when mode='woe'. "
                "WoE values are already on the log-odds scale."
            )


@dataclass
class ModelSpec:
    """
    A complete specification of a single candidate model.

    Parameters
    ----------
    name           : unique identifier used in tables, exports, and plots
    deal_variables : ordered list of deal variable names
    configs        : dict mapping each variable to its VariableConfig
    interactions   : subset of deal_variables that should be interacted
                     with the Equifax score
    description    : free-text notes (shown in exports for documentation)
    """
    name:           str
    deal_variables: List[str]
    configs:        Dict[str, VariableConfig]
    interactions:   List[str] = field(default_factory=list)
    description:    str       = ""

    def __post_init__(self) -> None:
        missing = [v for v in self.deal_variables if v not in self.configs]
        if missing:
            raise ValueError(
                f"[{self.name}] No VariableConfig provided for: {missing}"
            )
        bad_interactions = [
            v for v in self.interactions if v not in self.deal_variables
        ]
        if bad_interactions:
            raise ValueError(
                f"[{self.name}] Interaction variables must be in "
                f"deal_variables. Offenders: {bad_interactions}"
            )


# ---------------------------------------------------------------------------
# Per-model results container
# ---------------------------------------------------------------------------

@dataclass
class ModelResults:
    """Container for everything produced for a single model."""
    spec: ModelSpec

    # Fitted model + preprocessing artifacts (needed to re-score new data)
    sm_model:       object                    = None
    design_columns: List[str]                 = field(default_factory=list)
    equifax_scaler: Optional[StandardScaler]  = None
    cont_scalers:   Dict[str, StandardScaler] = field(default_factory=dict)
    cont_shifts:    Dict[str, float]          = field(default_factory=dict)

    # Coefficient diagnostics
    coefficient_table: Optional[pd.DataFrame] = None

    # Predictions and targets on dev and OOT
    y_dev:     Optional[pd.Series] = None
    y_oot:     Optional[pd.Series] = None
    pd_dev:    Optional[pd.Series] = None
    pd_oot:    Optional[pd.Series] = None
    score_dev: Optional[pd.Series] = None
    score_oot: Optional[pd.Series] = None

    # Metrics
    discrimination_dev: Optional[Dict]         = None
    discrimination_oot: Optional[Dict]         = None
    calibration_dev:    Optional[Dict]         = None
    calibration_oot:    Optional[Dict]         = None
    psi:                Optional[Dict]         = None
    csi:                Optional[pd.DataFrame] = None

    # Analytical views
    decile_dev:             Optional[pd.DataFrame] = None
    decile_oot:             Optional[pd.DataFrame] = None
    score_distribution_dev: Optional[pd.DataFrame] = None
    score_distribution_oot: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Main comparison class
# ---------------------------------------------------------------------------

class ModelComparison:
    """
    Orchestrates fitting and evaluating multiple model specifications.

    Steps
    -----
        1. fit(dev_df)              — fit every spec on the development sample
        2. evaluate(dev_df, oot_df) — score and compute metrics on both samples
        3. export_excel(path)       — PBI-ready long-format Excel
        4. export_html(path)        — self-contained Plotly dashboard
        5. summary()                — one-row-per-model headline table
    """

    # Scorecard scaling constants (matches existing ScorecardScaler defaults)
    PDO        = 20
    BASE_SCORE = 600
    BASE_ODDS  = 50

    def __init__(
        self,
        specs: List[ModelSpec],
        target: str,
        equifax_col: str,
    ):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"ModelSpec names must be unique. Got: {names}")

        self.specs       = specs
        self.target      = target
        self.equifax_col = equifax_col
        self.results: Dict[str, ModelResults] = {
            s.name: ModelResults(spec=s) for s in specs
        }

        # Derived scaling constants for PD → score conversion
        self._B = self.PDO / np.log(2)
        self._A = self.BASE_SCORE + self._B * np.log(self.BASE_ODDS)

    # ==================================================================
    # Step 1: Fit
    # ==================================================================

    def fit(self, dev_df: pd.DataFrame) -> "ModelComparison":
        """Fit each model spec on the development sample."""
        for spec in self.specs:
            self._fit_one(spec, dev_df)
        return self

    def _fit_one(self, spec: ModelSpec, dev_df: pd.DataFrame) -> None:
        result = self.results[spec.name]

        # Standardise Equifax on dev — store the scaler for OOT re-use
        eq_scaler = StandardScaler()
        equifax_std = eq_scaler.fit_transform(
            dev_df[[self.equifax_col]].values
        ).ravel()
        result.equifax_scaler = eq_scaler

        # Build deal-variable inputs (WoE or transformed-continuous)
        deal_inputs: Dict[str, np.ndarray] = {}
        for var in spec.deal_variables:
            cfg = spec.configs[var]
            deal_inputs[var] = self._prepare_deal_input(
                var, cfg, dev_df, result, fit=True,
            )

        # Assemble design matrix
        design_df = pd.DataFrame({"equifax": equifax_std}, index=dev_df.index)
        for var in spec.deal_variables:
            col_name = self._input_col_name(var, spec.configs[var])
            design_df[col_name] = deal_inputs[var]
        for var in spec.interactions:
            col_name = self._interaction_col_name(var, spec.configs[var])
            design_df[col_name] = equifax_std * deal_inputs[var]

        result.design_columns = list(design_df.columns)

        # Fit logistic regression
        y = dev_df[self.target].values
        X = sm.add_constant(design_df.values, has_constant="add")
        result.sm_model = sm.Logit(y, X).fit(disp=0)

        # Coefficient diagnostics
        result.coefficient_table = self._build_coefficient_table(
            result.sm_model, result.design_columns, design_df.values,
        )

    # ------------------------------------------------------------------
    # Design-matrix helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _input_col_name(var: str, cfg: VariableConfig) -> str:
        return f"woe_{var}" if cfg.mode == "woe" else f"cont_{var}"

    @staticmethod
    def _interaction_col_name(var: str, cfg: VariableConfig) -> str:
        base = ModelComparison._input_col_name(var, cfg)
        return f"int_eq_x_{base}"

    def _prepare_deal_input(
        self,
        var: str,
        cfg: VariableConfig,
        df: pd.DataFrame,
        result: ModelResults,
        fit: bool,
    ) -> np.ndarray:
        """
        Build the numeric column for a single deal variable, following
        its VariableConfig. When fit=True, scalers / shifts are stored
        on `result`; when fit=False, the stored ones are reused.
        """
        if cfg.mode == "woe":
            col = f"{var}_woe"
            if col not in df.columns:
                raise KeyError(
                    f"Expected WoE column '{col}' for variable '{var}'. "
                    "Run BinningPipeline.transform() before fitting."
                )
            return df[col].values.astype(float)

        # Continuous mode — apply transform, then standardise
        if var not in df.columns:
            raise KeyError(
                f"Expected raw column '{var}' in dataframe (continuous mode)."
            )
        raw = df[var].values.astype(float)
        transformed, shift = self._apply_transform(
            raw, cfg.transform, fit, result, var,
        )
        if fit:
            scaler = StandardScaler()
            standardised = scaler.fit_transform(
                transformed.reshape(-1, 1)
            ).ravel()
            result.cont_scalers[var] = scaler
            result.cont_shifts[var]  = shift
        else:
            scaler = result.cont_scalers[var]
            standardised = scaler.transform(
                transformed.reshape(-1, 1)
            ).ravel()
        return standardised

    @staticmethod
    def _apply_transform(
        values: np.ndarray,
        transform: str,
        fit: bool,
        result: ModelResults,
        var: str,
    ) -> Tuple[np.ndarray, float]:
        """
        Apply a named transformation and return (transformed_values, shift).

        For log:  shift = max(0, 1 - min(x)) so that log(x + shift) is finite.
        For sqrt: shift = max(0, -min(x)) so that the input is non-negative.

        At fit time the shift is computed from dev and stored on the result.
        At score time the stored shift is reused, keeping the transformation
        consistent across samples.
        """
        if transform == "none":
            return values, 0.0

        if fit:
            if transform == "log":
                shift = max(0.0, 1.0 - float(np.nanmin(values)))
            else:  # sqrt
                shift = max(0.0, -float(np.nanmin(values)))
        else:
            shift = result.cont_shifts[var]

        if transform == "log":
            return np.log(values + shift), shift
        return np.sqrt(values + shift), shift

    def _build_coefficient_table(
        self,
        sm_model,
        design_columns: List[str],
        X_no_const: np.ndarray,
    ) -> pd.DataFrame:
        """Coefficient summary including p-values, CIs and VIF."""
        all_names = ["const"] + design_columns
        params    = np.asarray(sm_model.params).astype(float)
        pvalues   = np.asarray(sm_model.pvalues).astype(float)
        std_errs  = np.asarray(sm_model.bse).astype(float)
        conf_int  = np.asarray(sm_model.conf_int()).astype(float)

        # VIF on the design (without the constant)
        vifs: List[Optional[float]] = [None]  # placeholder for const
        if X_no_const.shape[1] >= 2:
            for i in range(X_no_const.shape[1]):
                try:
                    vifs.append(float(variance_inflation_factor(X_no_const, i)))
                except Exception:
                    vifs.append(np.nan)
        else:
            vifs.append(np.nan)

        return pd.DataFrame({
            "term":        all_names,
            "coefficient": params,
            "std_error":   std_errs,
            "p_value":     pvalues,
            "conf_low":    conf_int[:, 0],
            "conf_high":   conf_int[:, 1],
            "vif":         vifs,
        })

    # ==================================================================
    # Step 2: Evaluate
    # ==================================================================

    def evaluate(
        self,
        dev_df: pd.DataFrame,
        oot_df: pd.DataFrame,
    ) -> "ModelComparison":
        """Score and compute all metrics on both samples for every spec."""
        for spec in self.specs:
            self._evaluate_one(spec, dev_df, oot_df)
        return self

    def _evaluate_one(
        self,
        spec: ModelSpec,
        dev_df: pd.DataFrame,
        oot_df: pd.DataFrame,
    ) -> None:
        result = self.results[spec.name]

        # Cache targets on the result so plot builders can reach them
        result.y_dev = dev_df[self.target].reset_index(drop=True)
        result.y_oot = oot_df[self.target].reset_index(drop=True)

        # Score both samples
        result.pd_dev = pd.Series(
            self._predict_proba(spec, dev_df), index=dev_df.index, name="pd",
        )
        result.pd_oot = pd.Series(
            self._predict_proba(spec, oot_df), index=oot_df.index, name="pd",
        )
        result.score_dev = self._pd_to_score(result.pd_dev)
        result.score_oot = self._pd_to_score(result.pd_oot)

        y_dev = dev_df[self.target]
        y_oot = oot_df[self.target]

        # Discrimination
        result.discrimination_dev = DiscriminationMetrics(
            y_dev, result.pd_dev, label="Development",
        ).summary()
        result.discrimination_oot = DiscriminationMetrics(
            y_oot, result.pd_oot, label="OOT",
        ).summary()

        # Calibration (both samples — useful for drift comparison)
        result.calibration_dev = self._calibration_block(y_dev, result.pd_dev)
        result.calibration_oot = self._calibration_block(y_oot, result.pd_oot)

        # Stability — PSI on scores, CSI on the design columns
        result.psi = StabilityMetrics.calculate_psi(
            expected = result.score_dev,
            actual   = result.score_oot,
            label    = "Score PSI",
        )

        design_dev = self._design_matrix(spec, dev_df)
        design_oot = self._design_matrix(spec, oot_df)
        result.csi = StabilityMetrics.run_csi_all(
            expected_df = design_dev,
            actual_df   = design_oot,
            variables   = list(design_dev.columns),
        )

        # Decile and distribution views
        result.decile_dev = self._decile_view(y_dev, result.score_dev, "Development")
        result.decile_oot = self._decile_view(y_oot, result.score_oot, "OOT")
        result.score_distribution_dev = self._score_distribution(
            y_dev, result.score_dev, "Development",
        )
        result.score_distribution_oot = self._score_distribution(
            y_oot, result.score_oot, "OOT",
        )

    @staticmethod
    def _calibration_block(y: pd.Series, p: pd.Series) -> Dict:
        calib = CalibrationMetrics(y, p)
        hl = calib.hosmer_lemeshow()
        return {
            "hl_statistic":     hl["statistic"],
            "hl_p_value":       hl["p_value"],
            "well_calibrated":  hl["well_calibrated"],
            "group_stats":      hl["group_stats"],
            "observed_vs_pred": calib.observed_vs_predicted(),
        }

    def _predict_proba(self, spec: ModelSpec, df: pd.DataFrame) -> np.ndarray:
        design = self._design_matrix(spec, df)
        result = self.results[spec.name]
        X = sm.add_constant(design.values, has_constant="add")
        return np.asarray(result.sm_model.predict(X))

    def _design_matrix(
        self,
        spec: ModelSpec,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rebuild the design matrix for scoring, reusing stored artifacts."""
        result = self.results[spec.name]
        equifax_std = result.equifax_scaler.transform(
            df[[self.equifax_col]].values
        ).ravel()

        deal_inputs: Dict[str, np.ndarray] = {}
        for var in spec.deal_variables:
            deal_inputs[var] = self._prepare_deal_input(
                var, spec.configs[var], df, result, fit=False,
            )

        design = pd.DataFrame({"equifax": equifax_std}, index=df.index)
        for var in spec.deal_variables:
            design[self._input_col_name(var, spec.configs[var])] = deal_inputs[var]
        for var in spec.interactions:
            col = self._interaction_col_name(var, spec.configs[var])
            design[col] = equifax_std * deal_inputs[var]
        return design

    def _pd_to_score(self, pd_values: pd.Series) -> pd.Series:
        """Convert PD to integer scorecard score via A - B·log-odds."""
        clipped = pd_values.clip(lower=1e-6, upper=1 - 1e-6)
        log_odds = np.log(clipped / (1 - clipped))
        return (self._A - self._B * log_odds).round().astype(int).rename("score")

    # ------------------------------------------------------------------
    # Decile + score-distribution views
    # ------------------------------------------------------------------

    @staticmethod
    def _decile_view(
        y: pd.Series,
        scores: pd.Series,
        label: str,
        n_deciles: int = 10,
    ) -> pd.DataFrame:
        """
        Decile-level monitoring view, ordered worst → best score
        (decile 1 = lowest score = highest risk).
        """
        df = pd.DataFrame({"y": y.values, "score": scores.values})
        # Ascending of -score reverses the order so decile 1 = worst
        df["decile"] = pd.qcut(
            -df["score"], q=n_deciles, labels=False, duplicates="drop",
        ) + 1

        grouped = (
            df.groupby("decile")
            .agg(
                n_total   = ("y", "count"),
                n_bads    = ("y", "sum"),
                min_score = ("score", "min"),
                max_score = ("score", "max"),
            )
            .reset_index()
            .sort_values("decile")
        )
        grouped["n_goods"]        = grouped["n_total"] - grouped["n_bads"]
        grouped["bad_rate"]       = grouped["n_bads"]  / grouped["n_total"]
        grouped["pct_bads"]       = grouped["n_bads"]  / grouped["n_bads"].sum()
        grouped["pct_goods"]      = grouped["n_goods"] / grouped["n_goods"].sum()
        grouped["cum_pct_bads"]   = grouped["pct_bads"].cumsum()
        grouped["cum_pct_goods"]  = grouped["pct_goods"].cumsum()
        grouped["ks"]             = (
            grouped["cum_pct_bads"] - grouped["cum_pct_goods"]
        ).abs()
        grouped.insert(0, "dataset", label)
        return grouped

    @staticmethod
    def _score_distribution(
        y: pd.Series,
        scores: pd.Series,
        label: str,
        n_bins: int = 25,
    ) -> pd.DataFrame:
        """Score histogram split by class — for goods vs bads overlay plots."""
        df = pd.DataFrame({"y": y.values, "score": scores.values})
        edges = np.linspace(df["score"].min(), df["score"].max() + 1, n_bins + 1)
        df["bin"] = pd.cut(df["score"], bins=edges, include_lowest=True)

        rows: List[Dict] = []
        for class_val, class_label in [(0, "Good"), (1, "Bad")]:
            subset = df[df["y"] == class_val]
            counts = subset["bin"].value_counts().sort_index()
            for bin_label, n in counts.items():
                rows.append({
                    "dataset": label,
                    "class":   class_label,
                    "bin":     str(bin_label),
                    "bin_mid": float((bin_label.left + bin_label.right) / 2),
                    "n":       int(n),
                })
        return pd.DataFrame(rows)

    # ==================================================================
    # Summary
    # ==================================================================

    def summary(self) -> pd.DataFrame:
        """Headline metrics — one row per model."""
        rows: List[Dict] = []
        for spec in self.specs:
            r = self.results[spec.name]
            if r.discrimination_dev is None:
                continue
            d_dev = r.discrimination_dev
            d_oot = r.discrimination_oot
            gini_drop = d_dev["gini"] - d_oot["gini"]
            rows.append({
                "model":        spec.name,
                "n_variables":  len(spec.deal_variables),
                "interactions": ", ".join(spec.interactions) or "—",
                "gini_dev":     d_dev["gini"],
                "gini_oot":     d_oot["gini"],
                "gini_drop":    round(gini_drop, 4),
                "ks_dev":       d_dev["ks_statistic"],
                "ks_oot":       d_oot["ks_statistic"],
                "hl_p_dev":     r.calibration_dev["hl_p_value"]
                                if r.calibration_dev else None,
                "hl_p_oot":     r.calibration_oot["hl_p_value"]
                                if r.calibration_oot else None,
                "score_psi":    r.psi["psi"]    if r.psi else None,
                "psi_status":   r.psi["status"] if r.psi else None,
            })
        return (
            pd.DataFrame(rows)
            .sort_values("gini_oot", ascending=False)
            .reset_index(drop=True)
        )

    # ==================================================================
    # Scorecard view (works for continuous + WoE modes uniformly)
    # ==================================================================

    DEFAULT_PERCENTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    def scorecard_long(
        self,
        dev_df: pd.DataFrame,
        percentiles: Tuple[float, ...] = DEFAULT_PERCENTILES,
    ) -> pd.DataFrame:
        """
        Per-model, per-term scorecard view.

        For each term in each model, shows the points contribution at a
        set of percentiles of the term's standardised input distribution
        on the development sample. The total points at a given percentile
        is *per term*, not a summed example customer — co-occurring
        percentiles across variables would require a joint distribution.

        Score = A − B·log-odds, so the points contribution of a term
        with coefficient β at input value x is −β·x·B. The intercept
        contributes a base shift that is added to BASE_SCORE.
        """
        rows: List[Dict] = []
        for spec in self.specs:
            r = self.results[spec.name]
            if r.sm_model is None:
                continue
            design = self._design_matrix(spec, dev_df)
            params = pd.Series(
                np.asarray(r.sm_model.params).astype(float),
                index=["const"] + r.design_columns,
            )

            # Base shift from intercept
            base_shift = -float(params["const"]) * self._B
            rows.append({
                "model":       spec.name,
                "term":        "const (base shift)",
                "percentile":  None,
                "input_value": None,
                "coefficient": float(params["const"]),
                "points":      round(base_shift, 2),
            })

            # Per-term points at each requested percentile
            for term in r.design_columns:
                coef = float(params[term])
                qs = design[term].quantile(list(percentiles)).to_dict()
                for pct in percentiles:
                    x = float(qs[pct])
                    points = -coef * x * self._B
                    rows.append({
                        "model":       spec.name,
                        "term":        term,
                        "percentile":  pct,
                        "input_value": round(x, 4),
                        "coefficient": round(coef, 4),
                        "points":      round(points, 2),
                    })
        return pd.DataFrame(rows)

    # ==================================================================
    # Step 3: Excel export (Power BI-friendly long format)
    # ==================================================================

    def export_excel(
        self,
        path: Union[str, Path],
        scorecard_dev_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """
        Write a multi-sheet long-format Excel file optimised for Power BI.

        Parameters
        ----------
        path             : destination path
        scorecard_dev_df : optional dev DataFrame for the 'scorecard' sheet.
                           Pass the same df used in fit() to get a per-term
                           points view by percentile. If None, the sheet
                           is omitted.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sheets: Dict[str, pd.DataFrame] = {
            "summary":             self.summary(),
            "model_specs":         self._specs_long(),
            "coefficients":        self._coefficients_long(),
            "discrimination":      self._discrimination_long(),
            "calibration_groups":  self._calibration_long(),
            "stability_psi":       self._psi_long(),
            "stability_csi":       self._csi_long(),
            "decile_analysis":     self._decile_long(),
            "score_distribution":  self._score_distribution_long(),
        }
        if scorecard_dev_df is not None:
            sheets["scorecard"] = self.scorecard_long(scorecard_dev_df)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    def _specs_long(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for spec in self.specs:
            for var in spec.deal_variables:
                cfg = spec.configs[var]
                rows.append({
                    "model":         spec.name,
                    "description":   spec.description,
                    "variable":      var,
                    "mode":          cfg.mode,
                    "transform":     cfg.transform,
                    "is_interacted": var in spec.interactions,
                })
        return pd.DataFrame(rows)

    def _coefficients_long(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for spec in self.specs:
            tbl = self.results[spec.name].coefficient_table
            if tbl is None:
                continue
            tbl = tbl.copy()
            tbl.insert(0, "model", spec.name)
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _discrimination_long(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for spec in self.specs:
            r = self.results[spec.name]
            for dataset, block in [
                ("Development", r.discrimination_dev),
                ("OOT",         r.discrimination_oot),
            ]:
                if block is None:
                    continue
                for metric in ("auc", "gini", "ks_statistic"):
                    rows.append({
                        "model":   spec.name,
                        "dataset": dataset,
                        "metric":  metric,
                        "value":   block[metric],
                    })
        return pd.DataFrame(rows)

    def _calibration_long(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for spec in self.specs:
            r = self.results[spec.name]
            for dataset, block in [
                ("Development", r.calibration_dev),
                ("OOT",         r.calibration_oot),
            ]:
                if block is None:
                    continue
                g = block["group_stats"].copy()
                g.insert(0, "model",   spec.name)
                g.insert(1, "dataset", dataset)
                frames.append(g)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _psi_long(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for spec in self.specs:
            r = self.results[spec.name]
            if not r.psi:
                continue
            d = r.psi["bin_detail"].copy()
            d.insert(0, "model",       spec.name)
            d.insert(1, "overall_psi", r.psi["psi"])
            d.insert(2, "status",      r.psi["status"])
            frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _csi_long(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for spec in self.specs:
            r = self.results[spec.name]
            if r.csi is None:
                continue
            d = r.csi.copy()
            d.insert(0, "model", spec.name)
            frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _decile_long(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for spec in self.specs:
            r = self.results[spec.name]
            for d in (r.decile_dev, r.decile_oot):
                if d is None:
                    continue
                d = d.copy()
                d.insert(0, "model", spec.name)
                frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _score_distribution_long(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for spec in self.specs:
            r = self.results[spec.name]
            for d in (r.score_distribution_dev, r.score_distribution_oot):
                if d is None:
                    continue
                d = d.copy()
                d.insert(0, "model", spec.name)
                frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ==================================================================
    # Step 4: HTML export — interactive Plotly prototype
    # ==================================================================

    def export_html(self, path: Union[str, Path]) -> Path:
        """
        Generate a single self-contained HTML file with interactive
        Plotly charts. Models are colour-coded consistently across
        charts and traces can be toggled via the legend.
        """
        try:
            import plotly.graph_objects as go  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Plotly is required for export_html(). "
                "Install with: pip install plotly"
            ) from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sections: List[str] = [self._html_header(), self._html_summary_section()]
        chart_blocks = [
            ("ROC Curves — Development vs OOT",     self._fig_roc(),
             "Each curve shows ranking ability. Curves further from the "
             "diagonal mean better discrimination. Dev and OOT are shown "
             "together so divergence stands out."),
            ("Calibration — Observed vs Predicted", self._fig_calibration(),
             "Points on the diagonal mean predicted PD matches the observed "
             "default rate. Systematic deviation suggests recalibration."),
            ("Bad Rate by Decile",                  self._fig_decile_bad_rate(),
             "Decile 1 is the worst score band. A clean monotonic decline "
             "indicates the model ranks risk correctly."),
            ("Cumulative % Bads Captured",          self._fig_cum_bads(),
             "The classic gains chart. Steeper early curves mean more bads "
             "are concentrated in the worst score deciles."),
            ("Score Distribution — Goods vs Bads",  self._fig_score_distribution(),
             "Greater separation between Good and Bad distributions means "
             "stronger discrimination at the score level."),
            ("Coefficient Significance",            self._fig_coefficients(),
             "P-values per term, one bar group per model. Bars under the "
             "red 0.05 line are statistically significant."),
            ("Stability — PSI and CSI",             self._fig_stability(),
             "PSI is overall score drift dev → OOT. CSI breaks the same "
             "calculation down by input feature. Orange = 0.10, red = 0.25."),
        ]
        for title, fig, note in chart_blocks:
            sections.append(self._html_chart_section(title, fig, note))
        sections.append("</body></html>")

        path.write_text("\n".join(sections), encoding="utf-8")
        return path

    # ------ HTML scaffold helpers ------

    @staticmethod
    def _html_header() -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PD Scorecard — Model Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: -apple-system, system-ui, "Segoe UI", Arial, sans-serif;
         margin: 0; padding: 24px; background: #fafafa; color: #1a1a1a; }
  h1   { font-size: 24px; margin: 0 0 4px 0; }
  h2   { font-size: 18px; margin: 36px 0 8px 0; color: #2c3e50;
         border-bottom: 1px solid #d0d0d0; padding-bottom: 4px; }
  .lede { color: #555; margin: 0 0 24px 0; font-size: 14px; }
  .note { font-size: 13px; color: #555; margin: 4px 0 14px 0; }
  .chart { background: #fff; border: 1px solid #e2e2e2; border-radius: 4px;
           padding: 12px; margin-bottom: 12px; }
  table { border-collapse: collapse; font-size: 13px; background: #fff;
          border: 1px solid #d0d0d0; }
  th, td { padding: 6px 12px; border-bottom: 1px solid #ececec; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f0f3f7; font-weight: 600; }
</style>
</head>
<body>
<h1>PD Scorecard — Model Comparison</h1>
<p class="lede">Interactive prototype. Click legend items to toggle traces.
Models are colour-coded consistently across charts.</p>
"""

    def _html_summary_section(self) -> str:
        df = self.summary()
        if df.empty:
            return "<p>No models evaluated yet.</p>"
        for col in df.select_dtypes(include="float").columns:
            df[col] = df[col].round(4)
        return (
            "<h2>Headline Summary</h2>"
            "<p class='note'>Sorted by OOT Gini descending. PSI status "
            "flags overall score drift dev → OOT.</p>"
            + df.to_html(index=False, border=0)
        )

    @staticmethod
    def _html_chart_section(title: str, fig, note: str) -> str:
        chart_html = fig.to_html(
            full_html=False, include_plotlyjs=False, div_id=None,
        )
        return (
            f"<h2>{title}</h2>"
            f"<p class='note'>{note}</p>"
            f"<div class='chart'>{chart_html}</div>"
        )

    # ------ Plotly figure builders ------

    @staticmethod
    def _model_colour(idx: int) -> str:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]
        return palette[idx % len(palette)]

    def _fig_roc(self):
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve

        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="lightgray"))
        for i, spec in enumerate(self.specs):
            r = self.results[spec.name]
            if r.pd_dev is None:
                continue
            colour = self._model_colour(i)
            for sample, y, p, dash, gini in [
                ("Dev", r.y_dev, r.pd_dev, "solid",
                 r.discrimination_dev["gini"]),
                ("OOT", r.y_oot, r.pd_oot, "dot",
                 r.discrimination_oot["gini"]),
            ]:
                fpr, tpr, _ = roc_curve(y.values, p.values)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{spec.name} ({sample}, Gini={gini:.3f})",
                    line=dict(color=colour, dash=dash),
                ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=480, margin=dict(l=50, r=20, t=20, b=50),
            legend=dict(orientation="v", yanchor="bottom", y=0.02,
                        xanchor="right", x=0.98),
        )
        return fig

    def _fig_calibration(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="lightgray"))
        max_val = 0.0
        for i, spec in enumerate(self.specs):
            r = self.results[spec.name]
            if r.calibration_dev is None:
                continue
            colour = self._model_colour(i)
            for sample, block, dash in [
                ("Dev", r.calibration_dev, "solid"),
                ("OOT", r.calibration_oot, "dot"),
            ]:
                if block is None:
                    continue
                obs = block["observed_vs_pred"]
                fig.add_trace(go.Scatter(
                    x=obs["predicted_pd"], y=obs["observed_bad_rate"],
                    mode="lines+markers",
                    name=f"{spec.name} ({sample})",
                    line=dict(color=colour, dash=dash),
                ))
                max_val = max(max_val,
                              float(obs["predicted_pd"].max()),
                              float(obs["observed_bad_rate"].max()))
        rng = [0, max_val * 1.05] if max_val > 0 else None
        fig.update_layout(
            xaxis_title="Predicted PD (decile mean)",
            yaxis_title="Observed Bad Rate",
            xaxis=dict(range=rng), yaxis=dict(range=rng),
            height=480, margin=dict(l=50, r=20, t=20, b=50),
        )
        return fig

    def _fig_decile_bad_rate(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        for i, spec in enumerate(self.specs):
            r = self.results[spec.name]
            if r.decile_dev is None:
                continue
            colour = self._model_colour(i)
            for sample, df, dash in [
                ("Dev", r.decile_dev, "solid"),
                ("OOT", r.decile_oot, "dot"),
            ]:
                fig.add_trace(go.Scatter(
                    x=df["decile"], y=df["bad_rate"],
                    mode="lines+markers",
                    name=f"{spec.name} ({sample})",
                    line=dict(color=colour, dash=dash),
                ))
        fig.update_layout(
            xaxis_title="Decile (1 = worst score)",
            yaxis_title="Observed Bad Rate",
            height=420, margin=dict(l=50, r=20, t=20, b=50),
            xaxis=dict(tickmode="linear"),
        )
        return fig

    def _fig_cum_bads(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="lightgray"))
        for i, spec in enumerate(self.specs):
            r = self.results[spec.name]
            if r.decile_dev is None:
                continue
            colour = self._model_colour(i)
            for sample, df, dash in [
                ("Dev", r.decile_dev, "solid"),
                ("OOT", r.decile_oot, "dot"),
            ]:
                cum_pop = df["n_total"].cumsum() / df["n_total"].sum()
                fig.add_trace(go.Scatter(
                    x=cum_pop, y=df["cum_pct_bads"],
                    mode="lines+markers",
                    name=f"{spec.name} ({sample})",
                    line=dict(color=colour, dash=dash),
                ))
        fig.update_layout(
            xaxis_title="Cumulative % Population (worst → best)",
            yaxis_title="Cumulative % Bads Captured",
            height=420, margin=dict(l=50, r=20, t=20, b=50),
        )
        return fig

    def _fig_score_distribution(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_models = len(self.specs)
        fig = make_subplots(
            rows=n_models, cols=2,
            subplot_titles=[
                f"{s.name} — {sample}"
                for s in self.specs for sample in ("Dev", "OOT")
            ],
            shared_xaxes=True,
            vertical_spacing=max(0.04, 0.18 / max(n_models, 1)),
            horizontal_spacing=0.06,
        )
        for i, spec in enumerate(self.specs):
            r = self.results[spec.name]
            if r.score_distribution_dev is None:
                continue
            for j, df in enumerate(
                [r.score_distribution_dev, r.score_distribution_oot]
            ):
                for cls, colour in [("Good", "#2ca02c"), ("Bad", "#d62728")]:
                    sub = df[df["class"] == cls]
                    fig.add_trace(
                        go.Bar(
                            x=sub["bin_mid"], y=sub["n"],
                            name=cls, marker_color=colour,
                            showlegend=(i == 0 and j == 0),
                            legendgroup=cls,
                        ),
                        row=i + 1, col=j + 1,
                    )
        fig.update_layout(
            barmode="overlay",
            height=300 * n_models,
            margin=dict(l=50, r=20, t=40, b=50),
            bargap=0,
        )
        fig.update_traces(opacity=0.6)
        fig.update_xaxes(title_text="Score")
        fig.update_yaxes(title_text="Count")
        return fig

    def _fig_coefficients(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        for i, spec in enumerate(self.specs):
            r = self.results[spec.name]
            if r.coefficient_table is None:
                continue
            tbl = r.coefficient_table
            fig.add_trace(go.Bar(
                x=tbl["term"], y=tbl["p_value"],
                name=spec.name, marker_color=self._model_colour(i),
            ))
        fig.add_hline(y=0.05, line_dash="dash", line_color="red")
        fig.update_layout(
            barmode="group",
            yaxis_title="p-value",
            xaxis_title="Term",
            height=420, margin=dict(l=50, r=20, t=20, b=80),
        )
        fig.update_xaxes(tickangle=-30)
        return fig

    def _fig_stability(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Overall Score PSI", "CSI by Feature"],
            column_widths=[0.3, 0.7],
        )
        model_names: List[str] = []
        psis: List[float] = []
        colours: List[str] = []
        for i, spec in enumerate(self.specs):
            r = self.results[spec.name]
            if r.psi is None:
                continue
            model_names.append(spec.name)
            psis.append(r.psi["psi"])
            colour = self._model_colour(i)
            colours.append(colour)
            if r.csi is not None and not r.csi.empty:
                fig.add_trace(go.Bar(
                    x=r.csi["variable"], y=r.csi["csi"],
                    name=spec.name, marker_color=colour, legendgroup=spec.name,
                ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=model_names, y=psis, name="Score PSI",
            marker_color=colours, showlegend=False,
        ), row=1, col=1)
        for threshold, line_colour in [(0.10, "orange"), (0.25, "red")]:
            fig.add_hline(y=threshold, line_dash="dash",
                          line_color=line_colour, row=1, col=1)
            fig.add_hline(y=threshold, line_dash="dash",
                          line_color=line_colour, row=1, col=2)
        fig.update_yaxes(title_text="PSI", row=1, col=1)
        fig.update_yaxes(title_text="CSI", row=1, col=2)
        fig.update_layout(
            height=420, margin=dict(l=50, r=20, t=40, b=80), barmode="group",
        )
        fig.update_xaxes(tickangle=-30, row=1, col=2)
        return fig