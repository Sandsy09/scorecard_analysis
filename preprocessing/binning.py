"""
preprocessing/binning.py

Handles variable binning, Weight of Evidence (WoE) calculation,
and Information Value (IV) for scorecard development.

Key classes:
    VariableBinner  — bins a single variable and calculates WoE/IV
    BinningPipeline — orchestrates binning across all model variables
                      and produces a WoE-transformed dataset

Theory recap:
    WoE_i  = ln(Distribution of Events_i / Distribution of Non-Events_i)
    IV     = sum[(pct_bads_i - pct_goods_i) * WoE_i]

    WoE > 0 → bin has more bads than expected (higher risk)
    WoE < 0 → bin has fewer bads than expected (lower risk)
    WoE = 0 → bin is uninformative

IV thresholds:
    < 0.02         : Useless
    0.02 – 0.10    : Weak
    0.10 – 0.30    : Medium
    0.30 – 0.50    : Strong
    > 0.50         : Suspiciously strong — check for data leakage
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Custom exception
# --------------------------------------------------------------------------

class BinningError(Exception):
    pass


# --------------------------------------------------------------------------
# Single variable binner
# --------------------------------------------------------------------------

class VariableBinner:
    """
    Bins a single variable and calculates WoE and IV.

    Supports:
        - Equal frequency binning (starting point for manual refinement)
        - Manual binning via explicit cut points
        - Categorical binning (grouping by category WoE similarity)

    After fitting, call .transform() to replace raw values with WoE values.
    """

    # Guardrails for bin validation
    MIN_BIN_PCT   = 0.05   # each bin must be >= 5% of population
    MIN_BIN_BADS  = 50     # each bin must have >= 50 bad observations

    def __init__(
        self,
        variable: str,
        target: str,
        variable_type: str = "continuous",
    ):
        """
        Parameters
        ----------
        variable      : name of the predictor column
        target        : name of the binary target column (1=bad, 0=good)
        variable_type : 'continuous' or 'categorical'
        """
        self.variable      = variable
        self.target        = target
        self.variable_type = variable_type

        # Set after fitting
        self.cut_points:      Optional[List[float]] = None
        self.woe_map:         Optional[Dict]        = None
        self.iv:              Optional[float]       = None
        self.bin_stats:       Optional[pd.DataFrame] = None
        self.validation_issues: List[str]           = []

    # ------------------------------------------------------------------
    # Fitting methods
    # ------------------------------------------------------------------

    def fit_equal_frequency(
        self, df: pd.DataFrame, n_bins: int = 10
    ) -> "VariableBinner":
        """
        Fit using equal frequency (quantile) bins.
        Use this as a starting point, then refine with fit_manual().
        """
        self._validate_inputs(df)

        if self.variable_type == "categorical":
            return self._fit_categorical(df)

        work = df[[self.variable, self.target]].copy()

        try:
            work["bin"], retbins = pd.qcut(
                work[self.variable], q=n_bins,
                duplicates="drop", retbins=True
            )
            # Store cut points so transform() can recreate the same bins
            self.cut_points = list(retbins[1:-1])
        except Exception as e:
            raise BinningError(
                f"Equal frequency binning failed for '{self.variable}': {e}"
            )

        return self._compute_woe_iv(work)

    def fit_manual(
        self, df: pd.DataFrame, cut_points: List[float]
    ) -> "VariableBinner":
        """
        Fit using explicitly provided cut points.

        Example
        -------
        binner.fit_manual(df, cut_points=[60, 80, 100])
        # Creates bins: (-inf, 60], (60, 80], (80, 100], (100, inf)
        """
        if self.variable_type == "categorical":
            raise BinningError(
                "Use fit_equal_frequency for categorical variables; "
                "merge categories via the woe_map after fitting."
            )

        self._validate_inputs(df)
        self.cut_points = sorted(cut_points)

        work = df[[self.variable, self.target]].copy()
        work["bin"] = pd.cut(
            work[self.variable],
            bins=[-np.inf] + self.cut_points + [np.inf],
        )

        return self._compute_woe_iv(work)

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        """
        Validate the dataframe before fitting.

        Checks:
            - variable column exists in df
            - target column exists in df
            - target is binary (contains only 0 and 1)
            - variable column is not entirely null
            - dataframe has enough rows to bin meaningfully
        """
        # Column presence
        if self.variable not in df.columns:
            raise BinningError(
                f"Variable '{self.variable}' not found in dataframe. "
                f"Available columns: {df.columns.tolist()}"
            )
        if self.target not in df.columns:
            raise BinningError(
                f"Target '{self.target}' not found in dataframe. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Binary target check
        unique_target = set(df[self.target].dropna().unique())
        if not unique_target.issubset({0, 1}):
            raise BinningError(
                f"Target '{self.target}' must be binary (0/1). "
                f"Found values: {unique_target}"
            )

        # All-null variable check
        if df[self.variable].isna().all():
            raise BinningError(
                f"Variable '{self.variable}' is entirely null — cannot bin."
            )

        # Minimum row count
        if len(df) < 100:
            warnings.warn(
                f"[{self.variable}] Dataframe has only {len(df)} rows. "
                "WoE estimates may be unstable on small samples."
            )

    def _fit_categorical(self, df: pd.DataFrame) -> "VariableBinner":
        work = df[[self.variable, self.target]].copy()
        work["bin"] = work[self.variable].fillna("Missing").astype(str)
        return self._compute_woe_iv(work)

    # ------------------------------------------------------------------
    # Core WoE / IV calculation
    # ------------------------------------------------------------------

    def _compute_woe_iv(self, work: pd.DataFrame) -> "VariableBinner":
        total_bads  = work[self.target].sum()
        total_goods = len(work) - total_bads

        if total_bads == 0 or total_goods == 0:
            raise BinningError(
                f"'{self.variable}': dataset must contain both bads and goods."
            )

        rows = []
        for bin_label, grp in work.groupby("bin", observed=True):
            n_total = len(grp)
            n_bads  = grp[self.target].sum()
            n_goods = n_total - n_bads

            # Protect against log(0) — flag and apply small correction
            if n_bads == 0 or n_goods == 0:
                warnings.warn(
                    f"[{self.variable}] Bin '{bin_label}' has zero bads or goods. "
                    "Consider merging with an adjacent bin. "
                    "Applying 0.5 correction for now."
                )
                n_bads  = max(n_bads,  0.5)
                n_goods = max(n_goods, 0.5)

            pct_bads   = n_bads  / total_bads
            pct_goods  = n_goods / total_goods
            woe        = np.log(pct_bads / pct_goods)
            iv_contrib = (pct_bads - pct_goods) * woe

            rows.append({
                "bin":            bin_label,
                "n_total":        n_total,
                "n_bads":         n_bads,
                "n_goods":        n_goods,
                "pct_population": n_total / len(work),
                "pct_bads":       pct_bads,
                "pct_goods":      pct_goods,
                "bad_rate":       n_bads / n_total,
                "woe":            woe,
                "iv_contribution": iv_contrib,
            })

        self.bin_stats = pd.DataFrame(rows)
        self.iv        = self.bin_stats["iv_contribution"].sum()

        # Build woe_map with keys that match what transform() will produce.
        #
        # The bug this fixes: pd.qcut during fitting creates interval objects
        # whose outer boundaries are the actual data min/max, e.g.:
        #     (1234.5, 25000.0]
        #
        # But transform() calls pd.cut with [-inf] + cut_points + [inf],
        # producing intervals like:
        #     (-inf, 25000.0]
        #
        # These are different objects — map() finds no matching key and
        # returns NaN for every value, triggering the unmapped warning.
        #
        # Fix: for continuous variables, rebuild the map using
        # pd.IntervalIndex.from_breaks with the canonical boundaries so
        # the keys exactly match what pd.cut produces in transform().
        # Categorical variables use string keys — unaffected.
        if self.variable_type == "categorical" or self.cut_points is None:
            self.woe_map = dict(
                zip(self.bin_stats["bin"], self.bin_stats["woe"])
            )
        else:
            breaks          = [-np.inf] + self.cut_points + [np.inf]
            canonical_index = pd.IntervalIndex.from_breaks(breaks, closed="right")
            self.woe_map    = dict(
                zip(canonical_index, self.bin_stats["woe"].tolist())
            )

        self._run_validation()
        return self

    # ------------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------------

    def _run_validation(self) -> None:
        issues = []

        # Minimum bin size
        small = self.bin_stats[
            self.bin_stats["pct_population"] < self.MIN_BIN_PCT
        ]
        if not small.empty:
            issues.append(
                f"Bins below 5% of population: {small['bin'].tolist()}"
            )

        # Minimum bad count
        low_bads = self.bin_stats[
            self.bin_stats["n_bads"] < self.MIN_BIN_BADS
        ]
        if not low_bads.empty:
            issues.append(
                f"Bins with < {self.MIN_BIN_BADS} bads: "
                f"{low_bads['bin'].tolist()}"
            )

        # Monotonicity (continuous variables only)
        if self.variable_type == "continuous" and len(self.bin_stats) > 2:
            woe_vals = self.bin_stats["woe"].values
            ascending  = all(woe_vals[i] <= woe_vals[i+1]
                             for i in range(len(woe_vals) - 1))
            descending = all(woe_vals[i] >= woe_vals[i+1]
                             for i in range(len(woe_vals) - 1))
            if not (ascending or descending):
                issues.append(
                    "WoE is non-monotonic — consider adjusting cut points "
                    "or merging unstable bins."
                )

        # High IV warning
        if self.iv and self.iv > 0.5:
            issues.append(
                f"IV={self.iv:.3f} is very high — check for data leakage."
            )

        self.validation_issues = issues

    # ------------------------------------------------------------------
    # Transform: replace raw values with WoE
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Map raw variable values to their WoE bin values.
        Returns a Series of WoE values aligned to df's index.
        """
        if self.woe_map is None:
            raise BinningError(
                f"Binner for '{self.variable}' has not been fitted yet."
            )

        if self.variable_type == "categorical":
            mapped = (
                df[self.variable]
                .fillna("Missing")
                .astype(str)
                .map(self.woe_map)
            )
        else:
            # Use pd.cut with labels= to assign WoE values positionally.
            #
            # Why not binned.map(woe_map)?
            # pd.cut returns a Categorical series. In pandas 2.0+, calling
            # .map(dict) on a Categorical does not reliably match pd.Interval
            # objects used as dict keys — even when the intervals appear
            # identical — due to how pandas resolves Categorical lookups
            # internally. This causes every value to return NaN.
            #
            # Using labels= bypasses dict lookup entirely: WoE values are
            # assigned positionally to bins at cut time. bin_stats rows are
            # in sorted bin order (groupby on ordered Categorical preserves
            # order), and pd.cut assigns labels in the same sorted order,
            # so positional alignment is guaranteed.
            bins       = [-np.inf] + (self.cut_points or []) + [np.inf]
            woe_labels = self.bin_stats["woe"].tolist()
            mapped     = (
                pd.cut(
                    df[self.variable],
                    bins=bins,
                    labels=woe_labels,
                    ordered=False,
                )
                .astype(float)
            )

        # Flag unmapped values (new categories or out-of-range values)
        n_missing = mapped.isna().sum()
        if n_missing > 0:
            warnings.warn(
                f"[{self.variable}] {n_missing} values could not be mapped "
                "to a WoE bin — will be NaN. Check for unseen categories "
                "or out-of-range values."
            )

        return mapped

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        if self.bin_stats is None:
            raise BinningError("Binner not fitted.")
        out = self.bin_stats.copy()
        out.insert(0, "variable", self.variable)
        out["iv_total"]  = self.iv
        out["iv_rating"] = self.iv_rating
        return out

    @property
    def iv_rating(self) -> str:
        if self.iv is None:
            return "Not fitted"
        thresholds = [
            (0.02, "Useless"),
            (0.10, "Weak"),
            (0.30, "Medium"),
            (0.50, "Strong"),
        ]
        for threshold, label in thresholds:
            if self.iv < threshold:
                return label
        return "Suspiciously Strong — check for data leakage"


# --------------------------------------------------------------------------
# Multi-variable pipeline
# --------------------------------------------------------------------------

class BinningPipeline:
    """
    Orchestrates binning across all model variables.

    Usage
    -----
    pipeline = (
        BinningPipeline(target='default_flag')
        .add_variable('annual_income', n_bins=10)
        .add_variable('ltv_ratio', cut_points=[60, 80, 100])
        .add_variable('employment_status', variable_type='categorical')
    )
    df_woe = pipeline.fit_transform(df_train)
    df_oot_woe = pipeline.transform(df_oot)  # reuses same bins
    """

    def __init__(self, target: str):
        self.target  = target
        self._configs: Dict[str, dict] = {}
        self.iv_summary: Optional[pd.DataFrame] = None

    def add_variable(
        self,
        variable: str,
        variable_type: str = "continuous",
        cut_points: Optional[List[float]] = None,
        n_bins: int = 10,
    ) -> "BinningPipeline":
        """
        Register a variable for binning.
        Provide cut_points for manual binning, or leave None for
        equal-frequency auto-binning.
        """
        self._configs[variable] = {
            "binner":      VariableBinner(variable, self.target, variable_type),
            "cut_points":  cut_points,
            "n_bins":      n_bins,
            "fitted":      False,
        }
        return self

    def fit(self, df: pd.DataFrame) -> "BinningPipeline":
        for var, cfg in self._configs.items():
            binner = cfg["binner"]
            if cfg["cut_points"] is not None:
                binner.fit_manual(df, cfg["cut_points"])
            else:
                binner.fit_equal_frequency(df, cfg["n_bins"])
            cfg["fitted"] = True
        self._build_iv_summary()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted WoE mappings to df.
        Adds a new column '{variable}_woe' for each registered variable.
        Raw columns are preserved.
        """
        out = df.copy()
        for var, cfg in self._configs.items():
            if not cfg["fitted"]:
                raise BinningError(
                    f"'{var}' has not been fitted. Call fit() first."
                )
            out[f"{var}_woe"] = cfg["binner"].transform(df)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_selected_variables(
        self,
        min_iv: float = 0.10,
        max_iv: float = 0.50,
    ) -> List[str]:
        """
        Return variables whose IV falls within the acceptable range.
        Defaults: 0.10 (medium) to 0.50 (strong, before leakage concern).
        """
        if self.iv_summary is None:
            raise BinningError("Pipeline not fitted.")
        mask = (
            (self.iv_summary["iv"] >= min_iv) &
            (self.iv_summary["iv"] <= max_iv)
        )
        return self.iv_summary.loc[mask, "variable"].tolist()

    def get_bin_stats(self, variable: str) -> pd.DataFrame:
        if variable not in self._configs:
            raise BinningError(f"'{variable}' not registered in pipeline.")
        return self._configs[variable]["binner"].bin_stats

    def get_all_bin_stats(self) -> Dict[str, pd.DataFrame]:
        return {
            var: cfg["binner"].bin_stats
            for var, cfg in self._configs.items()
            if cfg["fitted"]
        }

    def _build_iv_summary(self) -> None:
        rows = []
        for var, cfg in self._configs.items():
            if cfg["fitted"]:
                b = cfg["binner"]
                rows.append({
                    "variable": var,
                    "iv":       b.iv,
                    "rating":   b.iv_rating,
                    "n_bins":   len(b.bin_stats),
                    "issues":   "; ".join(b.validation_issues) or "None",
                })
        self.iv_summary = (
            pd.DataFrame(rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def print_iv_summary(self) -> None:
        if self.iv_summary is None:
            print("Pipeline not fitted.")
            return
        print("\n" + "=" * 65)
        print("IV SUMMARY")
        print("=" * 65)
        print(f"{'Variable':<30} {'IV':>6}  {'Rating':<30} {'Issues'}")
        print("-" * 65)
        for _, row in self.iv_summary.iterrows():
            print(
                f"{row['variable']:<30} {row['iv']:>6.3f}  "
                f"{row['rating']:<30} {row['issues']}"
            )
        print("=" * 65)
