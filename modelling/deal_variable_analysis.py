"""
modelling/deal_variable_analysis.py

Exploratory analysis of the relationship between deal variables and
default in log-odds space, stratified by Equifax score band.

Purpose
-------
Before fitting interaction models, this module answers three questions:

    1. What is the shape of the log-odds relationship for each deal
       variable? (Linear, monotonic non-linear, U-shaped, flat?)
       → Informs whether WoE binning is appropriate or whether a
         variable transformation is needed first.

    2. Does the log-odds profile shape change across Equifax bands, or
       does only the level shift?
       → Shape change = genuine interaction (confirms BD findings).
       → Level shift only = multiplicative structure approximately holds.

    3. How much does adding interaction terms actually improve the model
       over a standard additive logistic regression?
       → Quantified via LR test, AIC difference, and Gini uplift.

Theory recap
-----------
WoE for bin i:
    WoE_i = ln(P(Events | bin_i) / P(NonEvents | bin_i))
           = ln(bad_rate_i / good_rate_i)          [simplified]

This is a log-odds transformation relative to the population. If the
empirical log-odds of default within each bin is approximately linear
against the bin midpoint (for continuous variables), then WoE binning
is capturing the relationship well and no further transformation is
needed.

Non-linearities to look for:
    - U-shape (low and high values both high risk) → split variable
      or use a quadratic term before WoE binning
    - Monotonic but non-linear → log/sqrt transformation then rebinn
    - Flat / non-monotonic → variable may not be independently
      predictive after controlling for other variables; check IV

Equifax band stratification:
    If log-odds profiles are parallel across bands (same shape,
    different intercept) → multiplicative structure approximately holds.
    If slopes or shapes diverge across bands → interaction is real.
    The WoE comparison adds a second lens: if WoE values within each
    bin diverge across Equifax bands, the bin is not capturing a
    consistent effect.

Classes
-------
    DealVariableLogOddsAnalysis  — main analysis class
    LogOddsResult                — dataclass holding per-variable results
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from typing import Dict, List, Optional, Tuple

# Import the interaction and base logistic models for with/without comparison
# These are resolved relative to the project package root — adjust if needed
try:
    from modelling.interaction_model import InteractionLogisticRegression
    from preprocessing.binning import BinningPipeline
except ImportError:
    # Allow standalone use without the full package installed
    InteractionLogisticRegression = None  # type: ignore
    BinningPipeline = None               # type: ignore

try:
    import statsmodels.api as sm
except ImportError:
    sm = None  # type: ignore


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LogOddsResult:
    """Holds the log-odds analysis output for a single deal variable."""
    variable:           str
    overall_log_odds:   pd.DataFrame          # log-odds per bin, full population
    band_log_odds:      pd.DataFrame          # log-odds per bin per Equifax band
    band_woe:           pd.DataFrame          # WoE per bin per Equifax band
    spearman_overall:   float                 # ρ between midpoint and log-odds
    spearman_by_band:   Dict[str, float]      # ρ per band
    monotonic:          bool                  # True if consistent direction
    linearity_flag:     str                   # "Linear", "Monotonic", "Non-linear"
    transformation_suggestion: str
    transform_type:     str                   # "none" | "log" | "sqrt" | "split"
    interaction_evidence:  str               # interpretation vs other bands
    warnings:           List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class DealVariableLogOddsAnalysis:
    """
    Analyses the log-odds relationship between deal variables and default,
    with Equifax score band stratification.

    Produces:
        - Empirical log-odds per bin (overall and per Equifax band)
        - WoE recalculated within each Equifax band
        - Spearman correlation to test monotonicity
        - Transformation suggestions based on observed shape
        - With vs without interaction model comparison

    Usage
    -----
        analyser = DealVariableLogOddsAnalysis(
            target      = "default_flag",
            equifax_col = "equifax_score",
            n_bands     = 4,
            band_labels = ["Sub-Prime", "Near-Prime", "Prime", "Super-Prime"],
        )

        results = analyser.run(
            df           = dev_df,
            deal_vars    = ["ltv_ratio", "loan_term_months", "deposit_pct"],
            cut_points   = {"ltv_ratio": [60, 80, 100]},
            n_bins       = 10,
        )

        analyser.print_report(results)
    """

    MIN_BAND_BADS = 20   # minimum bads in a band×bin cell for reliable log-odds

    def __init__(
        self,
        target:       str,
        equifax_col:  str,
        n_bands:      int        = 4,
        band_labels:  Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        target      : binary target column (1 = bad, 0 = good)
        equifax_col : raw Equifax score column
        n_bands     : number of Equifax score bands (equal-frequency)
        band_labels : optional labels for bands, low to high.
                      Defaults to ["Band_1", ..., "Band_n"].
        """
        self.target      = target
        self.equifax_col = equifax_col
        self.n_bands     = n_bands
        self.band_labels = band_labels or [f"Band_{i+1}" for i in range(n_bands)]

        if len(self.band_labels) != n_bands:
            raise ValueError(
                f"band_labels length ({len(self.band_labels)}) must match "
                f"n_bands ({n_bands})."
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        df:          pd.DataFrame,
        deal_vars:   List[str],
        cut_points:  Optional[Dict[str, List[float]]] = None,
        var_types:   Optional[Dict[str, str]]         = None,
        n_bins:      int                              = 10,
    ) -> Dict[str, LogOddsResult]:
        """
        Run the full log-odds analysis for all deal variables.

        Parameters
        ----------
        df          : development dataframe
        deal_vars   : list of deal variable column names
        cut_points  : {variable: [cut1, cut2, ...]} for manual binning.
                      Variables not listed use equal-frequency quantile bins.
        var_types   : {variable: 'categorical'} for categorical variables.
        n_bins      : number of equal-frequency bins for auto-binned variables

        Returns
        -------
        Dict of {variable_name: LogOddsResult}
        """
        cut_points = cut_points or {}
        var_types  = var_types  or {}

        # Assign Equifax bands
        df = df.copy()
        df["_eq_band"] = pd.qcut(
            df[self.equifax_col],
            q      = self.n_bands,
            labels = self.band_labels,
        )

        results: Dict[str, LogOddsResult] = {}

        for var in deal_vars:
            if var not in df.columns:
                warnings.warn(f"Variable '{var}' not found in dataframe — skipping.")
                continue

            vtype = var_types.get(var, "continuous")
            cp    = cut_points.get(var)

            # Bin the variable across the full dataframe
            df, bin_col = self._bin_variable(df, var, vtype, cp, n_bins)

            result = self._analyse_variable(df, var, bin_col, vtype)
            results[var] = result

            # Clean up temporary bin column
            df.drop(columns=[bin_col], inplace=True)

        return results

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def _bin_variable(
        self,
        df:      pd.DataFrame,
        var:     str,
        vtype:   str,
        cp:      Optional[List[float]],
        n_bins:  int,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Bin a variable and return the dataframe with the new bin column,
        plus the bin column name.
        """
        bin_col = f"_bin_{var}"
        df = df.copy()

        if vtype == "categorical":
            df[bin_col] = df[var].fillna("Missing").astype(str)
            return df, bin_col

        if cp is not None:
            bins           = [-np.inf] + sorted(cp) + [np.inf]
            df[bin_col]    = pd.cut(df[var], bins=bins)
        else:
            try:
                df[bin_col] = pd.qcut(df[var], q=n_bins, duplicates="drop")
            except Exception as e:
                warnings.warn(
                    f"Equal-frequency binning failed for '{var}': {e}. "
                    "Falling back to 5 bins."
                )
                df[bin_col] = pd.qcut(df[var], q=5, duplicates="drop")

        return df, bin_col

    # ------------------------------------------------------------------
    # Per-variable analysis
    # ------------------------------------------------------------------

    def _analyse_variable(
        self,
        df:      pd.DataFrame,
        var:     str,
        bin_col: str,
        vtype:   str,
    ) -> LogOddsResult:

        # 1. Overall log-odds per bin
        overall_lo = self._compute_log_odds(df, bin_col, label="overall")

        # 2. Log-odds and WoE per bin per Equifax band
        band_lo_rows:  List[pd.DataFrame] = []
        band_woe_rows: List[pd.DataFrame] = []

        for band in self.band_labels:
            band_df = df[df["_eq_band"] == band]
            if band_df.empty:
                continue

            lo = self._compute_log_odds(band_df, bin_col, label=band)
            lo.insert(0, "equifax_band", band)
            band_lo_rows.append(lo)

            woe = self._compute_woe(band_df, bin_col, label=band)
            woe.insert(0, "equifax_band", band)
            band_woe_rows.append(woe)

        band_log_odds = pd.concat(band_lo_rows, ignore_index=True) if band_lo_rows else pd.DataFrame()
        band_woe      = pd.concat(band_woe_rows, ignore_index=True) if band_woe_rows else pd.DataFrame()

        # 3. Spearman correlation
        sp_overall = self._spearman(overall_lo, vtype)
        sp_by_band = {
            band: self._spearman(
                band_log_odds[band_log_odds["equifax_band"] == band], vtype
            )
            for band in self.band_labels
        }

        # 4. Linearity assessment
        mono, linearity_flag = self._assess_linearity(overall_lo, vtype)

        # 5. Transformation suggestion
        transform_type = self._detect_transform_type(overall_lo, linearity_flag, vtype)
        transform_sug = self._transformation_suggestion(
            overall_lo, band_log_odds, linearity_flag, vtype
        )

        # 6. Interaction evidence from band comparison
        interact_evidence = self._interaction_evidence(
            band_log_odds, sp_by_band
        )

        # 7. Warnings
        warn_list = self._build_warnings(overall_lo, band_log_odds, sp_overall)

        return LogOddsResult(
            variable                  = var,
            overall_log_odds          = overall_lo,
            band_log_odds             = band_log_odds,
            band_woe                  = band_woe,
            spearman_overall          = sp_overall,
            spearman_by_band          = sp_by_band,
            monotonic                 = mono,
            linearity_flag            = linearity_flag,
            transformation_suggestion = transform_sug,
            transform_type             = transform_type,
            interaction_evidence      = interact_evidence,
            warnings                  = warn_list,
        )

    # ------------------------------------------------------------------
    # Log-odds calculation
    # ------------------------------------------------------------------

    def _compute_log_odds(
        self,
        df:      pd.DataFrame,
        bin_col: str,
        label:   str = "",
    ) -> pd.DataFrame:
        """
        Compute empirical log-odds of default per bin.

        Log-odds = ln(P(bad | bin) / P(good | bin))
                 = ln(n_bads / n_goods)  [per bin]

        Returns DataFrame: bin | n_total | n_bads | n_goods | bad_rate |
                           log_odds | bin_midpoint
        """
        rows = []
        for bin_val, grp in df.groupby(bin_col, observed=True):
            n_total = len(grp)
            n_bads  = int(grp[self.target].sum())
            n_goods = n_total - n_bads

            # Apply 0.5 correction to avoid log(0); flag the cell
            sparse = n_bads == 0 or n_goods == 0
            n_bads_adj  = max(n_bads,  0.5)
            n_goods_adj = max(n_goods, 0.5)
            log_odds    = float(np.log(n_bads_adj / n_goods_adj))

            # Bin midpoint (for continuous bins only)
            midpoint: Optional[float] = None
            if hasattr(bin_val, "mid"):
                midpoint = float(bin_val.mid)

            rows.append({
                "bin":         bin_val,
                "n_total":     n_total,
                "n_bads":      n_bads,
                "n_goods":     n_goods,
                "bad_rate":    round(n_bads / n_total, 4) if n_total > 0 else np.nan,
                "log_odds":    round(log_odds, 4),
                "bin_midpoint": midpoint,
                "sparse":      sparse,
            })

        return pd.DataFrame(rows)

    def _compute_woe(
        self,
        df:      pd.DataFrame,
        bin_col: str,
        label:   str = "",
    ) -> pd.DataFrame:
        """
        Compute WoE per bin within the given dataframe subset.

        WoE_i = ln(pct_bads_i / pct_goods_i)

        This recalculates WoE within each Equifax band rather than using
        the overall WoE. Comparing these per-band WoE values shows whether
        the same bin has a consistent risk signal across credit quality tiers.
        """
        total_bads  = df[self.target].sum()
        total_goods = len(df) - total_bads

        if total_bads == 0 or total_goods == 0:
            return pd.DataFrame()

        rows = []
        for bin_val, grp in df.groupby(bin_col, observed=True):
            n_total = len(grp)
            n_bads  = int(grp[self.target].sum())
            n_goods = n_total - n_bads

            n_bads_adj  = max(n_bads,  0.5)
            n_goods_adj = max(n_goods, 0.5)

            pct_bads  = n_bads_adj  / total_bads
            pct_goods = n_goods_adj / total_goods
            woe       = float(np.log(pct_bads / pct_goods))
            iv_contrib = (pct_bads - pct_goods) * woe

            rows.append({
                "bin":        bin_val,
                "n_total":    n_total,
                "n_bads":     n_bads,
                "woe":        round(woe, 4),
                "iv_contrib": round(iv_contrib, 4),
                "bad_rate":   round(n_bads / n_total, 4) if n_total > 0 else np.nan,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Linearity and monotonicity
    # ------------------------------------------------------------------

    def _spearman(
        self, df: pd.DataFrame, vtype: str
    ) -> float:
        """Spearman ρ between bin midpoint and log-odds (continuous only)."""
        if vtype == "categorical" or df.empty:
            return np.nan
        valid = df[df["bin_midpoint"].notna() & df["log_odds"].notna()]
        if len(valid) < 3:
            return np.nan
        rho, _ = stats.spearmanr(valid["bin_midpoint"], valid["log_odds"])
        return round(float(rho), 4)

    def _assess_linearity(
        self, df: pd.DataFrame, vtype: str
    ) -> Tuple[bool, str]:
        """
        Classify the log-odds shape as Linear, Monotonic, or Non-linear.

        Method:
            1. Compute Spearman ρ — if |ρ| >= 0.90, classify as Monotonic.
            2. Additionally test Pearson r on midpoints vs log-odds —
               if both |ρ| and |r| >= 0.90, classify as Linear (well-captured
               by WoE binning with no further transformation needed).
            3. Otherwise Non-linear.
        """
        if vtype == "categorical" or df.empty:
            return True, "Categorical — not applicable"

        valid = df[df["bin_midpoint"].notna() & df["log_odds"].notna()]
        if len(valid) < 3:
            return True, "Insufficient bins to assess"

        mid     = valid["bin_midpoint"].values
        lo      = valid["log_odds"].values
        rho, _  = stats.spearmanr(mid, lo)
        r, _    = stats.pearsonr(mid, lo)

        # Monotonicity: check if direction is consistent across consecutive bins
        diffs    = np.diff(lo)
        n_pos    = int((diffs > 0).sum())
        n_neg    = int((diffs < 0).sum())
        n_steps  = len(diffs)
        dominant = max(n_pos, n_neg)
        monotonic = dominant >= (n_steps * 0.80)   # 80% of steps in one direction

        if abs(rho) >= 0.90 and abs(r) >= 0.90:
            return monotonic, "Linear"
        elif abs(rho) >= 0.75 or monotonic:
            return monotonic, "Monotonic"
        else:
            return False, "Non-linear"


    def _detect_transform_type(
        self,
        overall_lo:     pd.DataFrame,
        linearity_flag: str,
        vtype:          str,
    ) -> str:
        """
        Return a clean transform identifier used by the plotter.

        Returns
        -------
        "none"  — relationship is linear; no transformation required
        "log"   — log transformation recommended
        "sqrt"  — sqrt transformation recommended
        "split" — variable should be split at an inflection point
        "na"    — categorical or insufficient data
        """
        if vtype == "categorical" or linearity_flag in (
            "Categorical \u2014 not applicable", "Insufficient bins to assess"
        ):
            return "na"

        if linearity_flag == "Linear":
            return "none"

        if linearity_flag == "Monotonic":
            valid = overall_lo[overall_lo["bin_midpoint"].notna()].copy()
            if len(valid) >= 4 and (valid["bin_midpoint"].values > 0).all():
                from scipy import stats as _stats
                mid  = valid["bin_midpoint"].values
                lo   = valid["log_odds"].values
                r_lin,  _ = _stats.pearsonr(mid, lo)
                r_log,  _ = _stats.pearsonr(np.log(mid), lo)
                r_sqrt, _ = _stats.pearsonr(np.sqrt(mid), lo)
                if abs(r_log)  > abs(r_lin) + 0.05:
                    return "log"
                if abs(r_sqrt) > abs(r_lin) + 0.05:
                    return "sqrt"
            return "none"

        # Non-linear — check for U-shape or inverted-U
        valid = overall_lo[overall_lo["bin_midpoint"].notna()
                           & overall_lo["log_odds"].notna()]
        if len(valid) >= 3:
            lo_vals = valid["log_odds"].values
            min_idx = int(np.argmin(lo_vals))
            max_idx = int(np.argmax(lo_vals))
            n       = len(lo_vals)
            if (0 < min_idx < n - 1
                    and lo_vals[0] > lo_vals[min_idx]
                    and lo_vals[-1] > lo_vals[min_idx]):
                return "split"
            if (0 < max_idx < n - 1
                    and lo_vals[0] < lo_vals[max_idx]
                    and lo_vals[-1] < lo_vals[max_idx]):
                return "split"
        return "split"

    # ------------------------------------------------------------------
    # Transformation suggestions
    # ------------------------------------------------------------------

    def _transformation_suggestion(
        self,
        overall_lo:    pd.DataFrame,
        band_lo:       pd.DataFrame,
        linearity_flag: str,
        vtype:         str,
    ) -> str:
        """
        Suggest a transformation based on the observed log-odds shape.

        Logic:
            Linear        → no transformation needed; WoE binning is sufficient.
            Monotonic     → consider log or sqrt transformation to linearise
                            before WoE binning; this concentrates bins where
                            the relationship is steepest.
            Non-linear    → inspect shape:
                - U-shape (both tails high risk) → split into two variables
                  (below/above inflection point) or add a quadratic term.
                - Single peak (mid-values high risk) → same as U-shape.
                - Flat then steep (threshold effect) → restrict WoE bins to
                  capture the threshold; do not force monotonicity.
            Categorical   → review category groupings by WoE similarity;
                            merge categories with similar WoE.
        """
        if linearity_flag in ("Categorical — not applicable", "Insufficient bins to assess"):
            if vtype == "categorical":
                return (
                    "Categorical variable. Review WoE per category and group "
                    "categories with similar WoE together before modelling."
                )
            return "Insufficient data to make a suggestion."

        if linearity_flag == "Linear":
            return (
                "Log-odds relationship is approximately linear. "
                "WoE binning is appropriate with no further transformation needed."
            )

        if linearity_flag == "Monotonic":
            # Check if the relationship accelerates at the extremes
            valid = overall_lo[overall_lo["bin_midpoint"].notna()].copy()
            if len(valid) >= 4:
                lo_vals = valid["log_odds"].values
                mid_vals = valid["bin_midpoint"].values
                # Fit linear and log models; compare R²
                if (mid_vals > 0).all():
                    r_lin, _ = stats.pearsonr(mid_vals, lo_vals)
                    r_log, _ = stats.pearsonr(np.log(mid_vals), lo_vals)
                    if abs(r_log) > abs(r_lin) + 0.05:
                        return (
                            "Log-odds is monotonic but curves — log transformation of "
                            "the raw variable before WoE binning is likely to improve "
                            "bin stability and monotonicity. Apply log(variable) and rebin."
                        )
                    r_sqrt, _ = stats.pearsonr(np.sqrt(np.abs(mid_vals)), lo_vals)
                    if abs(r_sqrt) > abs(r_lin) + 0.05:
                        return (
                            "Log-odds is monotonic but curves — sqrt transformation "
                            "of the raw variable before WoE binning may help. "
                            "Apply sqrt(variable) and rebin."
                        )
            return (
                "Log-odds is monotonic but not perfectly linear. "
                "WoE binning will capture the direction correctly. "
                "If bin stability is poor, consider a log or sqrt transformation "
                "of the raw variable before rebinning."
            )

        # Non-linear
        valid = overall_lo[overall_lo["bin_midpoint"].notna() & overall_lo["log_odds"].notna()]
        if len(valid) >= 3:
            lo_vals = valid["log_odds"].values
            min_idx = int(np.argmin(lo_vals))
            max_idx = int(np.argmax(lo_vals))
            n       = len(lo_vals)

            # U-shape: minimum is in the middle, both ends higher
            if 0 < min_idx < n - 1 and (lo_vals[0] > lo_vals[min_idx]) and (lo_vals[-1] > lo_vals[min_idx]):
                return (
                    "U-shaped log-odds relationship detected — both low and high values "
                    "are high risk with a low-risk middle. "
                    "Action: split into two binary variables (below/above the low-risk trough) "
                    "or create a 'distance from optimum' feature before binning. "
                    "Do NOT force monotonicity in WoE binning."
                )

            # Inverted U: maximum in middle
            if 0 < max_idx < n - 1 and (lo_vals[0] < lo_vals[max_idx]) and (lo_vals[-1] < lo_vals[max_idx]):
                return (
                    "Inverted U-shaped log-odds — mid-range values are highest risk. "
                    "Action: split into below/above peak, or apply quadratic WoE binning. "
                    "Review whether this pattern makes business sense — may indicate "
                    "a confounding variable."
                )

        return (
            "Non-linear, non-monotonic log-odds relationship. "
            "Inspect the bin plot carefully. Consider splitting the variable at "
            "the inflection point(s) and treating each segment separately. "
            "Alternatively, apply a piecewise linear transformation before binning."
        )

    # ------------------------------------------------------------------
    # Interaction evidence from band comparison
    # ------------------------------------------------------------------

    def _interaction_evidence(
        self,
        band_lo:    pd.DataFrame,
        sp_by_band: Dict[str, float],
    ) -> str:
        """
        Assess whether the log-odds profile shape is consistent across
        Equifax bands or diverges, providing evidence for/against interaction.

        Two signals:
            1. Spearman ρ variance across bands — high variance suggests
               the direction or strength of the relationship changes
            2. Log-odds range per bin across bands — if the same bin has
               a very different log-odds in Band_1 vs Band_4, the interaction
               is real and the level shift alone cannot account for it
        """
        if band_lo.empty:
            return "No band data available."

        valid_rhos = [v for v in sp_by_band.values() if not np.isnan(v)]
        if not valid_rhos:
            return "Insufficient data per band to assess interaction."

        rho_range = max(valid_rhos) - min(valid_rhos)
        rho_std   = float(np.std(valid_rhos))

        # Per-bin log-odds range across bands
        if "log_odds" in band_lo.columns and "bin" in band_lo.columns:
            per_bin_range = (
                band_lo.groupby("bin", observed=True)["log_odds"]
                .apply(lambda x: x.max() - x.min())
            )
            mean_range = float(per_bin_range.mean()) if not per_bin_range.empty else 0.0
            max_range  = float(per_bin_range.max())  if not per_bin_range.empty else 0.0
        else:
            mean_range = 0.0
            max_range  = 0.0

        # Interpret
        if rho_range <= 0.15 and mean_range <= 0.30:
            return (
                f"Log-odds profiles are consistent across Equifax bands "
                f"(Spearman ρ range={rho_range:.2f}, mean per-bin log-odds range={mean_range:.2f}). "
                "The interaction appears to be primarily a level shift — "
                "multiplicative structure may approximately hold."
            )
        elif rho_range <= 0.30 and mean_range <= 0.60:
            return (
                f"Moderate divergence across Equifax bands "
                f"(Spearman ρ range={rho_range:.2f}, mean per-bin log-odds range={mean_range:.2f}). "
                "Some interaction is present. Review band-level plots to assess "
                "whether the difference is concentrated in specific bins or bands."
            )
        else:
            return (
                f"Strong divergence across Equifax bands "
                f"(Spearman ρ range={rho_range:.2f}, mean per-bin log-odds range={mean_range:.2f}, "
                f"max per-bin range={max_range:.2f}). "
                "The shape of the log-odds relationship changes materially across "
                "credit quality tiers — interaction terms are strongly justified."
            )

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _build_warnings(
        self,
        overall_lo: pd.DataFrame,
        band_lo:    pd.DataFrame,
        sp_overall: float,
    ) -> List[str]:
        warn: List[str] = []

        sparse_bins = overall_lo[overall_lo["sparse"]]["bin"].tolist()
        if sparse_bins:
            warn.append(
                f"⚠️  Bins with zero bads or goods (0.5 correction applied): "
                f"{sparse_bins}. Log-odds estimates in these bins are unreliable. "
                "Consider merging with adjacent bins."
            )

        if not np.isnan(sp_overall) and abs(sp_overall) < 0.30:
            warn.append(
                f"⚠️  Very low Spearman ρ ({sp_overall:.2f}) — the variable shows "
                "little monotonic relationship with log-odds overall. "
                "Review IV and consider whether this variable is independently "
                "predictive after controlling for Equifax score."
            )

        if not band_lo.empty:
            small_cells = band_lo[band_lo["n_bads"] < self.MIN_BAND_BADS]
            if not small_cells.empty:
                warn.append(
                    f"⚠️  {len(small_cells)} band×bin cells have < {self.MIN_BAND_BADS} bads. "
                    "Per-band log-odds estimates may be unstable. "
                    "Interpret stratified plots cautiously."
                )

        return warn

    # ------------------------------------------------------------------
    # With vs without interaction model comparison
    # ------------------------------------------------------------------

    def compare_interaction_models(
        self,
        df:          pd.DataFrame,
        deal_vars:   List[str],
        woe_cols:    Optional[List[str]] = None,
        cut_points:  Optional[Dict[str, List[float]]] = None,
        var_types:   Optional[Dict[str, str]]         = None,
        n_bins:      int                              = 10,
    ) -> pd.DataFrame:
        """
        Fit and compare two model variants per deal variable:

            Base model:        logit(PD) = β0 + β_eq·Equifax_std + β_j·WoE_j
            Interaction model: logit(PD) = β0 + β_eq·Equifax_std + β_j·WoE_j
                                               + β_ij·(Equifax_std × WoE_j)

        For each deal variable, reports:
            - AIC (base and interaction)
            - BIC (base and interaction)
            - ΔLL (log-likelihood improvement)
            - LR test statistic and p-value (1 df — the interaction term)
            - Whether the interaction term is significant
            - Gini on development sample for both models

        This directly quantifies the value of the interaction term
        for each deal variable independently, before combining them
        into a full model in InteractionScorecardPipeline.

        Parameters
        ----------
        df          : development dataframe (must contain equifax_col, target,
                      and either raw deal vars or pre-computed WoE columns)
        deal_vars   : list of deal variable names (raw, before WoE)
        woe_cols    : if WoE-transformed columns already exist in df, pass
                      them here as {variable: woe_col_name}.
                      If None, WoE binning is run internally.
        cut_points  : manual cut points for binning (if woe_cols not provided)
        var_types   : variable types for binning (if woe_cols not provided)
        n_bins      : auto-bin count (if woe_cols not provided)

        Returns
        -------
        DataFrame: variable | aic_base | aic_interaction | delta_aic |
                   bic_base | bic_interaction | delta_bic |
                   delta_ll | lr_stat | lr_p_value |
                   interaction_significant | gini_base | gini_interaction |
                   gini_uplift
        """
        if sm is None:
            raise ImportError(
                "statsmodels is required for compare_interaction_models()."
            )

        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        from scipy.stats import chi2

        # Standardise Equifax
        scaler     = StandardScaler()
        eq_std     = scaler.fit_transform(df[[self.equifax_col]]).flatten()
        y          = df[self.target].values

        # Run binning if WoE columns are not pre-supplied
        df_woe = df.copy()
        if woe_cols is None:
            if BinningPipeline is None:
                raise ImportError(
                    "BinningPipeline not available. Either install the scorecard "
                    "package or pass pre-computed WoE columns via woe_cols."
                )
            binner = BinningPipeline(self.target)
            for var in deal_vars:
                binner.add_variable(
                    var,
                    variable_type = (var_types or {}).get(var, "continuous"),
                    cut_points    = (cut_points or {}).get(var),
                    n_bins        = n_bins,
                )
            df_woe = binner.fit_transform(df_woe)

        rows = []
        for var in deal_vars:
            woe_col = f"{var}_woe"
            if woe_col not in df_woe.columns:
                warnings.warn(f"WoE column '{woe_col}' not found — skipping {var}.")
                continue

            woe_vals = df_woe[woe_col].values
            interact = eq_std * woe_vals

            # --- Base model: Equifax_std + WoE_j ---
            X_base        = pd.DataFrame({
                "equifax_std": eq_std,
                f"woe_{var}":  woe_vals,
            })
            X_base_c      = sm.add_constant(X_base)
            fit_base      = sm.Logit(y, X_base_c).fit(disp=0)
            y_pred_base   = fit_base.predict(X_base_c).values
            gini_base     = 2 * float(roc_auc_score(y, y_pred_base)) - 1

            # --- Interaction model: Equifax_std + WoE_j + Equifax_std×WoE_j ---
            X_inter       = pd.DataFrame({
                "equifax_std":          eq_std,
                f"woe_{var}":           woe_vals,
                f"equifax_x_{var}":     interact,
            })
            X_inter_c     = sm.add_constant(X_inter)
            fit_inter     = sm.Logit(y, X_inter_c).fit(disp=0)
            y_pred_inter  = fit_inter.predict(X_inter_c).values
            gini_inter    = 2 * float(roc_auc_score(y, y_pred_inter)) - 1

            # LR test: 1 df (the interaction term)
            delta_ll = fit_inter.llf - fit_base.llf
            lr_stat  = 2 * delta_ll
            lr_pval  = 1 - chi2.cdf(lr_stat, df=1)

            rows.append({
                "variable":                var,
                "aic_base":                round(fit_base.aic,  2),
                "aic_interaction":         round(fit_inter.aic, 2),
                "delta_aic":               round(fit_inter.aic - fit_base.aic, 2),
                "bic_base":                round(fit_base.bic,  2),
                "bic_interaction":         round(fit_inter.bic, 2),
                "delta_bic":               round(fit_inter.bic - fit_base.bic, 2),
                "delta_ll":                round(delta_ll, 4),
                "lr_stat":                 round(lr_stat,  4),
                "lr_p_value":              round(lr_pval,  4),
                "interaction_significant": bool(lr_pval < 0.05),
                "gini_base":               round(gini_base,  4),
                "gini_interaction":        round(gini_inter, 4),
                "gini_uplift":             round(gini_inter - gini_base, 4),
            })

        result_df = (
            pd.DataFrame(rows)
            .sort_values("lr_p_value")
            .reset_index(drop=True)
        )
        self._print_interaction_comparison(result_df)
        return result_df

    @staticmethod
    def _print_interaction_comparison(df: pd.DataFrame) -> None:
        sep = "=" * 72
        print(f"\n{sep}")
        print("  WITH vs WITHOUT INTERACTION — MODEL COMPARISON")
        print("  (One interaction term tested per deal variable independently)")
        print(sep)
        print(
            "\n  Δ AIC < 0 → interaction model better (lower AIC is better)"
            "\n  LR p < 0.05 → interaction term is statistically significant"
            "\n  Gini uplift → discriminatory power added by the interaction\n"
        )
        print(df.to_string(index=False))
        print(f"\n{sep}\n")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, "LogOddsResult"]) -> None:
        """
        Print the full log-odds analysis report for all variables.
        """
        sep = "=" * 72

        print(f"\n{sep}")
        print("  DEAL VARIABLE LOG-ODDS ANALYSIS")
        print(f"  Equifax bands: {self.band_labels}")
        print(sep)

        for var, res in results.items():
            print(f"\n\n{'─' * 72}")
            print(f"  VARIABLE: {var.upper()}")
            print(f"{'─' * 72}")

            print(f"\n  Overall Log-Odds per Bin")
            print("  " + "-" * 50)
            print(res.overall_log_odds.to_string(index=False))

            print(f"\n  Shape Assessment")
            print("  " + "-" * 50)
            print(f"  Linearity:               {res.linearity_flag}")
            print(f"  Monotonic:               {res.monotonic}")
            print(f"  Spearman ρ (overall):    {res.spearman_overall:.4f}"
                  if not np.isnan(res.spearman_overall) else
                  "  Spearman ρ (overall):    N/A (categorical or insufficient data)")
            print(f"  Transformation advice:   {res.transformation_suggestion}")

            print(f"\n  Spearman ρ by Equifax Band")
            print("  " + "-" * 50)
            for band, rho in res.spearman_by_band.items():
                rho_str = f"{rho:.4f}" if not np.isnan(rho) else "N/A"
                print(f"  {band:<20} ρ = {rho_str}")

            print(f"\n  WoE by Equifax Band")
            print("  " + "-" * 50)
            if not res.band_woe.empty:
                print(res.band_woe.to_string(index=False))
            else:
                print("  No band WoE data available.")

            print(f"\n  Log-Odds by Equifax Band")
            print("  " + "-" * 50)
            if not res.band_log_odds.empty:
                print(res.band_log_odds[
                    ["equifax_band", "bin", "n_total", "n_bads",
                     "bad_rate", "log_odds", "sparse"]
                ].to_string(index=False))
            else:
                print("  No band log-odds data available.")

            print(f"\n  Interaction Evidence")
            print("  " + "-" * 50)
            print(f"  {res.interaction_evidence}")

            if res.warnings:
                print(f"\n  Warnings")
                print("  " + "-" * 50)
                for w in res.warnings:
                    print(f"  {w}")

        print(f"\n{sep}\n")