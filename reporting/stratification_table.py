"""
reporting/stratification_table.py

Produces a per-stratum summary table for a single deal variable,
documenting the odds ratio, confidence interval, and segment-level
p-value for each customer risk band.

Intended use: supporting documentation for the abandonment of the
stratified modelling approach in favour of an explicit interaction
term model (PD = logReg(Equifax, Deal, Equifax × Deal, ...)).

The table mirrors the stratification logic in ScorecardPipeline
.run_interaction_testing() so results are consistent with the
Breslow-Day output already produced.

Per-stratum p-value uses Fisher's Exact Test rather than chi-square
because individual strata may contain sparse cells — Fisher's is
exact and does not rely on large-sample approximations.

Usage
-----
    from reporting.stratification_table import build_stratification_table

    table = build_stratification_table(
        df              = pipeline.dev_data,
        deal_var        = "ltv_ratio",
        strata_var      = "equifax_score",
        target          = "default_flag",
        n_strata        = 3,
        strata_labels   = ["Low Risk", "Medium Risk", "High Risk"],
    )
    print(table.to_string(index=False))
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional, Tuple

from testing.statistical_tests import ContingencyTable


def build_stratification_table(
    df: pd.DataFrame,
    deal_var: str,
    strata_var: str,
    target: str,
    n_strata: int = 3,
    strata_labels: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Build a per-stratum odds ratio summary table for a single deal variable.

    The deal variable is binarised at its median within the full dataset
    (high = above median), consistent with the approach used in
    ScorecardPipeline.run_interaction_testing(). This produces a 2×2
    contingency table per stratum from which the OR, CI, and p-value
    are derived.

    Parameters
    ----------
    df            : raw development DataFrame (pre-WoE transformation)
    deal_var      : name of the deal variable column to summarise
    strata_var    : continuous variable used to define customer risk strata
                   (e.g. equifax_score, credit_bureau_score)
    target        : binary target column (1 = default, 0 = no default)
    n_strata      : number of equal-frequency risk bands (default 3)
    strata_labels : ordered labels for each band, low → high risk.
                    Must have length == n_strata.
                    Defaults to ["Low", "Medium", "High"] for n_strata=3,
                    or "Band 1", "Band 2", ... for other values.
    alpha         : significance level for confidence intervals (default 0.05)

    Returns
    -------
    DataFrame with columns:
        stratification_band   : score range (e.g. "(520, 650]")
        stratification_label  : human-readable band label
        total_observations    : number of customers in band
        total_defaults        : number of default events in band
        default_rate          : observed default rate within band
        deal_factor_or        : odds ratio (high deal vs low deal within band)
        ci_lower              : lower bound of (1-alpha) confidence interval
        ci_upper              : upper bound of (1-alpha) confidence interval
        or_95ci               : formatted string "OR (lower – upper)" for reports
        p_value               : Fisher's Exact Test p-value for the 2×2 table
        sparse_flag           : True if any 2×2 cell < ContingencyTable threshold

    Raises
    ------
    ValueError : if required columns are missing, target is non-binary,
                 or strata_labels length does not match n_strata.
    """
    _validate_inputs(df, deal_var, strata_var, target, n_strata, strata_labels)

    labels: List[str] = _resolve_labels(n_strata, strata_labels)

    work = df[[deal_var, strata_var, target]].copy()

    # Binarise deal variable at dataset-level median — same as pipeline
    median_deal = work[deal_var].median()
    work["_deal_high"] = (work[deal_var] > median_deal).astype(int)

    # Equal-frequency stratification on strata_var — same as pipeline
    work["_stratum_label"], band_edges = pd.qcut(
        work[strata_var],
        q=n_strata,
        labels=labels,
        retbins=True,
    )

    # Build one row per stratum
    rows = []
    for label, band_low, band_high in zip(
        labels, band_edges[:-1], band_edges[1:]
    ):
        subset = work[work["_stratum_label"] == label]

        if subset.empty:
            continue

        band_str = _format_band(band_low, band_high, strata_var)

        n_total    = int(len(subset))
        n_defaults = int(subset[target].sum())
        default_rate = n_defaults / n_total if n_total > 0 else np.nan

        # 2×2 table: deal_high × target within this stratum
        a = int(((subset["_deal_high"] == 1) & (subset[target] == 1)).sum())
        b = int(((subset["_deal_high"] == 1) & (subset[target] == 0)).sum())
        c = int(((subset["_deal_high"] == 0) & (subset[target] == 1)).sum())
        d = int(((subset["_deal_high"] == 0) & (subset[target] == 0)).sum())

        table = ContingencyTable(a, b, c, d, stratum_name=label)

        or_val          = table.odds_ratio
        ci_lower, ci_upper = _safe_ci(table, alpha)
        p_value         = _fisher_exact_p(a, b, c, d)
        sparse_flag     = table.is_sparse
        or_ci_str       = _format_or_ci(or_val, ci_lower, ci_upper)

        rows.append({
            "stratification_band":  band_str,
            "stratification_label": label,
            "total_observations":   n_total,
            "total_defaults":       n_defaults,
            "default_rate":         round(default_rate, 4),
            "deal_factor_or":       round(or_val, 3) if np.isfinite(or_val) else np.nan,
            "ci_lower":             round(ci_lower, 3) if np.isfinite(ci_lower) else np.nan,
            "ci_upper":             round(ci_upper, 3) if np.isfinite(ci_upper) else np.nan,
            "or_95ci":              or_ci_str,
            "p_value":              round(p_value, 4),
            "sparse_flag":          sparse_flag,
        })

    result = pd.DataFrame(rows)
    _print_table(result, deal_var, strata_var, median_deal, alpha)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    df: pd.DataFrame,
    deal_var: str,
    strata_var: str,
    target: str,
    n_strata: int,
    strata_labels: Optional[List[str]],
) -> None:
    missing = [c for c in [deal_var, strata_var, target] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    unique_target = set(df[target].dropna().unique())
    if not unique_target.issubset({0, 1}):
        raise ValueError(
            f"Target '{target}' must be binary (0/1). Found: {unique_target}"
        )

    if strata_labels is not None and len(strata_labels) != n_strata:
        raise ValueError(
            f"strata_labels has {len(strata_labels)} entries but "
            f"n_strata={n_strata}. They must match."
        )


def _resolve_labels(
    n_strata: int,
    strata_labels: Optional[List[str]],
) -> List[str]:
    if strata_labels is not None:
        return strata_labels
    if n_strata == 3:
        return ["Low", "Medium", "High"]
    return [f"Band {i + 1}" for i in range(n_strata)]


def _format_band(low: float, high: float, var_name: str) -> str:
    """Format the score interval as a human-readable string."""
    low_str  = f"{low:,.0f}"  if abs(low)  < 1e10 else "-∞"
    high_str = f"{high:,.0f}" if abs(high) < 1e10 else "+∞"
    return f"({low_str}, {high_str}]"


def _safe_ci(
    table: ContingencyTable,
    alpha: float,
) -> Tuple[float, float]:
    """
    Return the confidence interval, falling back to (nan, nan) when
    the Woolf formula is undefined (zero cells in the 2×2 table).
    """
    if np.isnan(table.var_log_or) or not np.isfinite(table.log_odds_ratio):
        return np.nan, np.nan
    return table.confidence_interval(alpha=alpha)


def _fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher's Exact Test p-value for the 2×2 table.

    Fisher's Exact is used in preference to chi-square because individual
    strata may have sparse cells where the chi-square approximation breaks
    down. Fisher's is exact regardless of cell counts.
    """
    _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(p_value)


def _format_or_ci(or_val: float, ci_lower: float, ci_upper: float) -> str:
    """Return a report-ready 'OR (lower – upper)' string."""
    if not np.isfinite(or_val):
        return "Undefined (zero cell)"
    if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        return f"{or_val:.3f} (CI undefined — zero cell)"
    return f"{or_val:.3f} ({ci_lower:.3f} – {ci_upper:.3f})"


def _print_table(
    result: pd.DataFrame,
    deal_var: str,
    strata_var: str,
    median_deal: float,
    alpha: float,
) -> None:
    ci_pct = int((1 - alpha) * 100)

    print(f"\n{'=' * 75}")
    print(f"STRATIFICATION SUMMARY — Deal Variable: {deal_var}")
    print(f"  Stratified by: {strata_var}")
    print(f"  Deal binarised at median: {median_deal:.4f}  (1 = above median)")
    print(f"  OR = odds of default: high deal vs low deal within stratum")
    print(f"  CI: {ci_pct}% Woolf interval   |   p-value: Fisher's Exact (two-sided)")
    print(f"{'=' * 75}")

    display_cols = [
        "stratification_band",
        "stratification_label",
        "total_observations",
        "total_defaults",
        "default_rate",
        "or_95ci",
        "p_value",
        "sparse_flag",
    ]

    print(result[display_cols].to_string(index=False))

    sparse_count = result["sparse_flag"].sum()
    if sparse_count:
        print(
            f"\n  ⚠️  {sparse_count} stratum/strata flagged as sparse "
            f"(event count < {ContingencyTable.SPARSE_THRESHOLD}). "
            "OR estimates and CIs may be unreliable in these bands. "
            "Consider merging adjacent strata or applying Tarone correction "
            "to the Breslow-Day test."
        )
    else:
        print(f"\n  ✅  No sparse strata detected.")

    print(f"{'=' * 75}\n")