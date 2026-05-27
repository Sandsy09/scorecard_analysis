"""
testing/statistical_tests.py

Statistical tests for validating the PD_cust × f(deal) model structure.

Classes:
    ContingencyTable           — a single 2×2 table for one stratum
    BreslowDayTest             — tests homogeneity of odds ratios across strata
                                 (with optional Tarone correction for sparse data)
    CMHTest                    — Cochran-Mantel-Haenszel test of association
                                 after controlling for stratification
    InteractionTestingPipeline — orchestrates the full BD → CMH flow
                                 with decision logic for each outcome
    StratumDiagnostics         — granular 2×2 tables per stratum × bin for a
                                 single deal variable; used to investigate why
                                 Breslow-Day has failed
    DealVariableDiagnostics    — runs StratumDiagnostics across all deal variables
                                 and produces a triage summary

Theory recap:
    Breslow-Day: H0 = all odds ratios equal across strata
        p > 0.05 → homogeneous → multiplicative structure supported
        p < 0.05 → interaction present → investigate

    CMH (run only if BD passes):
        H0 = no association between deal variable and default
             after controlling for customer risk strata
        p < 0.05 → genuine predictive power → include variable

    Tarone correction: applied when any stratum has sparse cells (<50 bads).
        More conservative than standard BD — produces equal or larger p-value.
        Use when BD is borderline (p ~ 0.03-0.07) to check if result holds.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Enums and result dataclasses
# --------------------------------------------------------------------------

class TestDecision(Enum):
    PASS       = "PASS"
    FAIL       = "FAIL"
    BORDERLINE = "BORDERLINE — Manual Review Required"


@dataclass
class BreslowDayResult:
    statistic:          float
    p_value:            float
    degrees_of_freedom: int
    tarone_statistic:   Optional[float]
    tarone_p_value:     Optional[float]
    tarone_applied:     bool
    effective_p_value:  float           # whichever p-value was used for decision
    decision:           TestDecision
    common_odds_ratio:  float
    odds_ratios:        Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    driving_strata:     List[str]
    recommendation:     str
    actions:            List[str] = field(default_factory=list)


@dataclass
class CMHResult:
    statistic:          float
    p_value:            float
    common_odds_ratio:  float
    decision:           TestDecision
    recommendation:     str


@dataclass
class InteractionResult:
    variable:        str
    breslow_day:     BreslowDayResult
    cmh:             Optional[CMHResult]
    final_decision:  str
    actions:         List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 2×2 Contingency Table
# --------------------------------------------------------------------------

class ContingencyTable:
    """
    Represents a single 2×2 table for one stratum.

    Layout:
        |               | Default (Bad) | No Default (Good) |
        |---------------|---------------|-------------------|
        | High deal risk |      a        |        b          |
        | Low deal risk  |      c        |        d          |

    For example, when testing LTV:
        a = High LTV customers who defaulted
        b = High LTV customers who did not default
        c = Low LTV customers who defaulted
        d = Low LTV customers who did not default
    """

    SPARSE_THRESHOLD = 50  # minimum bads per cell for standard BD

    def __init__(
        self,
        a: float, b: float,
        c: float, d: float,
        stratum_name: str = "",
    ):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.stratum_name = stratum_name

        if self.n == 0:
            raise ValueError(f"Stratum '{stratum_name}' has zero observations.")

    @property
    def n(self) -> float:
        return self.a + self.b + self.c + self.d

    @property
    def row1_total(self) -> float:
        return self.a + self.b   # exposed (high deal risk)

    @property
    def row2_total(self) -> float:
        return self.c + self.d   # unexposed (low deal risk)

    @property
    def col1_total(self) -> float:
        return self.a + self.c   # defaults

    @property
    def col2_total(self) -> float:
        return self.b + self.d   # non-defaults

    @property
    def odds_ratio(self) -> float:
        denom = self.b * self.c
        if denom == 0:
            return np.inf
        return (self.a * self.d) / denom

    @property
    def log_odds_ratio(self) -> float:
        or_val = self.odds_ratio
        if or_val in (0, np.inf):
            return np.nan
        return np.log(or_val)

    @property
    def var_log_or(self) -> float:
        """Variance of the log odds ratio (Woolf formula)."""
        cells = [self.a, self.b, self.c, self.d]
        if any(c == 0 for c in cells):
            return np.nan
        return sum(1.0 / c for c in cells)

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        z      = stats.norm.ppf(1 - alpha / 2)
        log_or = self.log_odds_ratio
        se     = np.sqrt(self.var_log_or) if not np.isnan(self.var_log_or) else 0
        return (np.exp(log_or - z * se), np.exp(log_or + z * se))

    @property
    def is_sparse(self) -> bool:
        """True if any event cell is below the sparse threshold."""
        return min(self.a, self.c) < self.SPARSE_THRESHOLD


# --------------------------------------------------------------------------
# Breslow-Day Test
# --------------------------------------------------------------------------

class BreslowDayTest:
    """
    Tests whether the odds ratio between a deal variable and default
    is consistent (homogeneous) across customer risk strata.

    This directly validates the multiplicative model assumption:
        PD = PD_cust × f(deal)
    which requires the deal effect to be the same regardless of customer type.

    Tarone correction is applied automatically when sparse cells are detected,
    or can be forced with apply_tarone=True.
    """

    ALPHA_FAIL       = 0.05
    ALPHA_BORDERLINE = 0.07   # between 0.05-0.07 treated as borderline

    def __init__(
        self,
        tables: List[ContingencyTable],
        alpha: float = 0.05,
    ):
        if len(tables) < 2:
            raise ValueError("Breslow-Day requires at least 2 strata.")
        self.tables = tables
        self.alpha  = alpha

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self, apply_tarone: Optional[bool] = None
    ) -> BreslowDayResult:
        """
        Run the Breslow-Day test.

        Parameters
        ----------
        apply_tarone : bool or None
            None  → auto-detect (applied if any table is sparse)
            True  → always apply Tarone correction
            False → never apply Tarone correction
        """
        has_sparse = any(t.is_sparse for t in self.tables)
        if apply_tarone is None:
            apply_tarone = has_sparse

        # Estimate common odds ratio via Mantel-Haenszel
        common_or = self._mh_common_or()

        # Compute BD statistic
        bd_stat, expected_as, variance_as = self._bd_statistic(common_or)
        df_val    = len(self.tables) - 1
        p_value   = 1 - stats.chi2.cdf(bd_stat, df=df_val)

        # Tarone correction
        tarone_stat = tarone_p = None
        if apply_tarone:
            tarone_stat, tarone_p = self._tarone_correction(
                bd_stat, expected_as, variance_as
            )

        effective_p = tarone_p if (apply_tarone and tarone_p is not None) else p_value
        decision    = self._make_decision(effective_p)

        # Per-stratum ORs and CIs
        odds_ratios = {t.stratum_name: t.odds_ratio for t in self.tables}
        conf_ints   = {t.stratum_name: t.confidence_interval() for t in self.tables}

        driving_strata = self._find_driving_strata(odds_ratios, common_or)
        recommendation = self._recommendation(decision, driving_strata, has_sparse)
        actions        = self._actions(decision, driving_strata)

        return BreslowDayResult(
            statistic          = bd_stat,
            p_value            = p_value,
            degrees_of_freedom = df_val,
            tarone_statistic   = tarone_stat,
            tarone_p_value     = tarone_p,
            tarone_applied     = apply_tarone,
            effective_p_value  = effective_p,
            decision           = decision,
            common_odds_ratio  = common_or,
            odds_ratios        = odds_ratios,
            confidence_intervals = conf_ints,
            driving_strata     = driving_strata,
            recommendation     = recommendation,
            actions            = actions,
        )

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------

    def _mh_common_or(self) -> float:
        """Mantel-Haenszel estimate of the common odds ratio."""
        num   = sum(t.a * t.d / t.n for t in self.tables)
        denom = sum(t.b * t.c / t.n for t in self.tables)
        return num / denom if denom != 0 else np.inf

    def _expected_a(self, t: ContingencyTable, common_or: float) -> float:
        """
        Solve quadratic for the expected value of cell 'a'
        given the common OR and fixed marginal totals.
        """
        if common_or == 1.0:
            return t.row1_total * t.col1_total / t.n

        A = 1 - common_or
        B = (t.row1_total + t.col1_total) * (common_or - 1) - t.n
        C = common_or * t.row1_total * t.col1_total

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return t.row1_total * t.col1_total / t.n

        sqrt_d = np.sqrt(discriminant)
        roots  = [(-B + sqrt_d) / (2 * A), (-B - sqrt_d) / (2 * A)]

        # Pick the root that falls in the valid range for cell a
        valid  = [
            r for r in roots
            if 0 < r < min(t.row1_total, t.col1_total)
        ]
        return valid[0] if valid else t.row1_total * t.col1_total / t.n

    def _variance_a(self, t: ContingencyTable, e_a: float) -> float:
        """Variance of cell 'a' given expected value e_a."""
        r2 = t.row2_total
        c2 = t.col2_total
        denom = (
            1.0 / e_a +
            1.0 / (t.row1_total - e_a) +
            1.0 / (t.col1_total - e_a) +
            1.0 / (r2 - t.col1_total + e_a)
        )
        return 1.0 / denom if denom != 0 else 0.0

    def _bd_statistic(
        self, common_or: float
    ) -> Tuple[float, List[float], List[float]]:
        expected_as = [self._expected_a(t, common_or) for t in self.tables]
        variance_as = [
            self._variance_a(t, e_a)
            for t, e_a in zip(self.tables, expected_as)
        ]
        bd_stat = sum(
            (t.a - e_a) ** 2 / v_a
            for t, e_a, v_a in zip(self.tables, expected_as, variance_as)
            if v_a > 0
        )
        return bd_stat, expected_as, variance_as

    def _tarone_correction(
        self,
        bd_stat: float,
        expected_as: List[float],
        variance_as: List[float],
    ) -> Tuple[float, float]:
        """
        Apply Tarone (1985) correction to reduce over-dispersion
        in sparse tables. The corrected statistic is always <= BD statistic.
        """
        sum_obs_minus_exp = sum(
            t.a - e_a for t, e_a in zip(self.tables, expected_as)
        )
        sum_var = sum(variance_as)
        if sum_var == 0:
            return bd_stat, 1.0

        correction    = (sum_obs_minus_exp ** 2) / sum_var
        tarone_stat   = bd_stat - correction
        df_val        = len(self.tables) - 1
        tarone_p      = 1 - stats.chi2.cdf(max(tarone_stat, 0), df=df_val)
        return tarone_stat, tarone_p

    def _make_decision(self, p_value: float) -> TestDecision:
        if p_value > self.ALPHA_BORDERLINE:
            return TestDecision.PASS
        elif p_value > self.ALPHA_FAIL:
            return TestDecision.BORDERLINE
        return TestDecision.FAIL

    def _find_driving_strata(
        self, odds_ratios: Dict[str, float], common_or: float
    ) -> List[str]:
        """Flag strata whose OR deviates most from the common OR."""
        if common_or in (0, np.inf):
            return []
        deviations = {
            name: abs(np.log(max(or_val, 1e-9)) - np.log(common_or))
            for name, or_val in odds_ratios.items()
            if or_val not in (np.inf, 0)
        }
        if not deviations:
            return []
        threshold = 0.5 * max(deviations.values())
        return [name for name, dev in deviations.items() if dev >= threshold]

    def _recommendation(
        self,
        decision: TestDecision,
        driving_strata: List[str],
        has_sparse: bool,
    ) -> str:
        if decision == TestDecision.PASS:
            return (
                "Odds ratios are homogeneous across strata. "
                "The multiplicative model structure is supported. "
                "Proceed to CMH test."
            )
        elif decision == TestDecision.BORDERLINE:
            suffix = (
                " Tarone correction applied due to sparse cells." if has_sparse else ""
            )
            return (
                f"Borderline result (p between 0.05–0.07).{suffix} "
                f"Investigate strata: {driving_strata}. "
                "Assess whether the interaction is practically significant "
                "before deciding whether to act."
            )
        else:
            return (
                f"Significant interaction detected. "
                f"Strata driving the difference: {driving_strata}. "
                "Investigate whether this is a genuine interaction, a data "
                "artefact, or a sample size issue. "
                "See actions for next steps."
            )

    def _actions(
        self, decision: TestDecision, driving_strata: List[str]
    ) -> List[str]:
        if decision == TestDecision.PASS:
            return ["Proceed to CMH test."]
        elif decision == TestDecision.BORDERLINE:
            return [
                "Perform pairwise Breslow-Day to isolate which strata differ.",
                "Assess practical significance — are the ORs substantively different?",
                "Proceed with caution; document and monitor.",
            ]
        else:
            return [
                f"Investigate data quality in strata: {driving_strata}.",
                "Check whether interaction is real or driven by a confounding variable.",
                "Option A: If minor/practical difference is small → accept and document.",
                "Option B: If moderate → consider stratified modelling.",
                "Option C: If strong and explainable → add interaction term to model.",
                "If adding interaction term: validate on OOT sample; "
                "document interpretability trade-off for governance.",
            ]


# --------------------------------------------------------------------------
# Pairwise Breslow-Day (for identifying which strata differ)
# --------------------------------------------------------------------------

class PairwiseBreslowDay:
    """
    Runs Breslow-Day pairwise across all combinations of strata
    to identify which specific pair(s) are driving a global failure.
    """

    def __init__(self, tables: List[ContingencyTable], alpha: float = 0.05):
        self.tables = tables
        self.alpha  = alpha

    def run(self) -> pd.DataFrame:
        results = []
        for i in range(len(self.tables)):
            for j in range(i + 1, len(self.tables)):
                pair = [self.tables[i], self.tables[j]]
                bd   = BreslowDayTest(pair, alpha=self.alpha)
                res  = bd.run()
                results.append({
                    "stratum_a":   self.tables[i].stratum_name,
                    "stratum_b":   self.tables[j].stratum_name,
                    "or_a":        self.tables[i].odds_ratio,
                    "or_b":        self.tables[j].odds_ratio,
                    "bd_p_value":  res.effective_p_value,
                    "decision":    res.decision.value,
                })
        return pd.DataFrame(results)


# --------------------------------------------------------------------------
# CMH Test
# --------------------------------------------------------------------------

class CMHTest:
    """
    Cochran-Mantel-Haenszel test.

    After Breslow-Day confirms homogeneous odds ratios, CMH confirms
    that the deal variable genuinely predicts default after controlling
    for the customer risk stratum.

    H0: no association between deal variable and default, controlling for strata
    H1: consistent association exists across strata

    A significant result (p < 0.05) means the deal variable adds real
    predictive value beyond the customer component.
    """

    def __init__(self, tables: List[ContingencyTable], alpha: float = 0.05):
        self.tables = tables
        self.alpha  = alpha

    def run(self) -> CMHResult:
        # Observed vs expected defaults in the high-deal-risk group
        obs_sum = sum(t.a for t in self.tables)
        exp_sum = sum(
            t.row1_total * t.col1_total / t.n
            for t in self.tables
        )
        var_sum = sum(
            (t.row1_total * t.row2_total * t.col1_total * t.col2_total)
            / (t.n ** 2 * (t.n - 1))
            for t in self.tables if t.n > 1
        )

        if var_sum == 0:
            cmh_stat = 0.0
            p_value  = 1.0
        else:
            # Continuity-corrected CMH statistic
            cmh_stat = (abs(obs_sum - exp_sum) - 0.5) ** 2 / var_sum
            p_value  = 1 - stats.chi2.cdf(cmh_stat, df=1)

        # Mantel-Haenszel common odds ratio
        mh_num    = sum(t.a * t.d / t.n for t in self.tables)
        mh_denom  = sum(t.b * t.c / t.n for t in self.tables)
        common_or = mh_num / mh_denom if mh_denom != 0 else np.inf

        decision       = TestDecision.PASS if p_value < self.alpha else TestDecision.FAIL
        recommendation = self._recommendation(decision, common_or, p_value)

        return CMHResult(
            statistic         = cmh_stat,
            p_value           = p_value,
            common_odds_ratio = common_or,
            decision          = decision,
            recommendation    = recommendation,
        )

    def _recommendation(
        self, decision: TestDecision, common_or: float, p_value: float
    ) -> str:
        if decision == TestDecision.PASS:
            return (
                f"Significant association confirmed after controlling for customer risk "
                f"(p={p_value:.4f}, MH OR={common_or:.2f}). "
                "Variable has genuine independent predictive power. Include in model."
            )
        return (
            f"No significant association after controlling for strata "
            f"(p={p_value:.4f}). "
            "Variable may not add independent value beyond customer variables. "
            "Consider excluding or further investigating."
        )


# --------------------------------------------------------------------------
# Full testing pipeline with flow logic
# --------------------------------------------------------------------------

class InteractionTestingPipeline:
    """
    Orchestrates the full Breslow-Day → CMH decision flow for each variable.

    Flow:
        1. Run Breslow-Day (with auto Tarone correction if sparse)
        2a. If FAIL   → flag interaction, generate investigation actions
        2b. If BORDER → run CMH but flag for manual review
        2c. If PASS   → run CMH to confirm independent predictive power
        3. CMH PASS   → include variable
           CMH FAIL   → consider excluding variable
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha   = alpha
        self._results: Dict[str, InteractionResult] = {}

    def run_variable(
        self,
        variable_name: str,
        tables: List[ContingencyTable],
        apply_tarone: Optional[bool] = None,
    ) -> InteractionResult:

        # Step 1: Breslow-Day
        # apply_tarone is passed through directly — None means auto-detect
        # from the BD table cell counts. Pass True to force the correction
        # when StratumDiagnostics reveals WoE-bin-level sparsity that the
        # binary-split BD tables would not auto-detect.
        bd_result = BreslowDayTest(tables, alpha=self.alpha).run(
            apply_tarone=apply_tarone
        )

        cmh_result     = None
        final_decision = ""
        actions        = list(bd_result.actions)   # copy

        # Step 2: Flow logic
        if bd_result.decision == TestDecision.FAIL:
            final_decision = "INTERACTION_DETECTED — investigate before including"

        elif bd_result.decision == TestDecision.BORDERLINE:
            cmh_result     = CMHTest(tables, alpha=self.alpha).run()
            final_decision = "BORDERLINE — manual review required"
            actions.append(cmh_result.recommendation)

        else:  # PASS
            cmh_result = CMHTest(tables, alpha=self.alpha).run()
            if cmh_result.decision == TestDecision.PASS:
                final_decision = "INCLUDE VARIABLE"
            else:
                final_decision = "EXCLUDE VARIABLE — no independent predictive power"
            actions = [cmh_result.recommendation]

        result = InteractionResult(
            variable       = variable_name,
            breslow_day    = bd_result,
            cmh            = cmh_result,
            final_decision = final_decision,
            actions        = actions,
        )
        self._results[variable_name] = result
        return result

    def summary(self) -> pd.DataFrame:
        """Return a one-row-per-variable summary of all test results."""
        rows = []
        for var, res in self._results.items():
            bd  = res.breslow_day
            cmh = res.cmh
            rows.append({
                "variable":       var,
                "bd_statistic":   round(bd.statistic, 4),
                "bd_p_value":     round(bd.p_value, 4),
                "tarone_applied": bd.tarone_applied,
                "tarone_p_value": round(bd.tarone_p_value, 4) if bd.tarone_p_value else None,
                "bd_decision":    bd.decision.value,
                "cmh_statistic":  round(cmh.statistic, 4) if cmh else None,
                "cmh_p_value":    round(cmh.p_value, 4) if cmh else None,
                "mh_odds_ratio":  round(cmh.common_odds_ratio, 3) if cmh else None,
                "final_decision": res.final_decision,
            })
        return pd.DataFrame(rows)

    def print_results(self) -> None:
        print("\n" + "=" * 70)
        print("INTERACTION TESTING RESULTS")
        print("=" * 70)
        for var, res in self._results.items():
            bd  = res.breslow_day
            cmh = res.cmh
            print(f"\nVariable: {var}")
            print(f"  Breslow-Day: stat={bd.statistic:.3f}, "
                  f"p={bd.p_value:.4f} "
                  f"{'(Tarone: p=' + f'{bd.tarone_p_value:.4f})' if bd.tarone_applied else ''}"
                  f" → {bd.decision.value}")
            if cmh:
                print(f"  CMH:         stat={cmh.statistic:.3f}, "
                      f"p={cmh.p_value:.4f}, "
                      f"OR={cmh.common_odds_ratio:.2f} → {cmh.decision.value}")
            print(f"  Decision:    {res.final_decision}")
            if res.actions:
                print("  Actions:")
                for action in res.actions:
                    print(f"    → {action}")
        print("\n" + "=" * 70)
        print("SUMMARY")
        print(self.summary().to_string(index=False))
        print("=" * 70)


# --------------------------------------------------------------------------
# Stratum Diagnostics
# --------------------------------------------------------------------------

class StratumDiagnostics:
    """
    Produces granular 2×2 contingency tables for every stratum × bin
    combination of a single deal variable.

    This is the investigation step after a Breslow-Day FAIL. Rather than
    the single high/low split used in BD, this shows the full picture —
    one table per bin per stratum — so you can see exactly where the
    odds ratios are diverging and whether it's driven by sparse cells,
    zero cells, or a genuine interaction in a specific segment.

    Each 2×2 table compares:
        - Customers IN this bin vs OUT of this bin
        - Within a single customer risk stratum

    The odds ratio per row answers:
        "Within this stratum, how much more likely to default are
         customers in this bin vs all other bins?"

    If ORs are consistent across strata for a given bin → no interaction
    If ORs diverge across strata for a given bin → interaction is real
    """

    def __init__(self, min_cell_threshold: int = ContingencyTable.SPARSE_THRESHOLD):
        """
        Parameters
        ----------
        min_cell_threshold : cells below this count are flagged as sparse.
                             Defaults to ContingencyTable.SPARSE_THRESHOLD (50)
                             so that sparsity flags here are consistent with
                             what BreslowDayTest uses to trigger the Tarone
                             correction. Set to 5 only if you want to flag
                             cells that are sparse for chi-square purposes.
        """
        self.min_cell_threshold = min_cell_threshold

    def run(
        self,
        df: pd.DataFrame,
        deal_var: str,
        customer_var: str,
        outcome_var: str,
    ) -> pd.DataFrame:
        """
        Run diagnostics for a single deal variable.

        Parameters
        ----------
        df           : DataFrame containing all variables
        deal_var     : binned deal variable column (bin labels or WoE floats)
        customer_var : stratifying customer variable (already binned into strata)
        outcome_var  : binary target column (1 = bad, 0 = good)

        Returns
        -------
        DataFrame with one row per stratum × bin combination, containing
        the 2×2 cell counts, odds ratio, and sparsity flags.
        """
        rows = []
        strata = sorted(df[customer_var].dropna().unique())

        for stratum in strata:
            subset = df[df[customer_var] == stratum]
            bins   = sorted(subset[deal_var].dropna().unique())

            for bin_val in bins:
                in_bin  = subset[subset[deal_var] == bin_val]
                out_bin = subset[subset[deal_var] != bin_val]

                a = int((in_bin[outcome_var]  == 1).sum())  # bad,  in bin
                b = int((in_bin[outcome_var]  == 0).sum())  # good, in bin
                c = int((out_bin[outcome_var] == 1).sum())  # bad,  out of bin
                d = int((out_bin[outcome_var] == 0).sum())  # good, out of bin

                min_cell = min(a, b, c, d)
                has_zero = any(v == 0 for v in [a, b, c, d])

                # Odds ratio — nan if denominator is zero
                or_val = (a * d) / (b * c) if (b * c) != 0 else np.nan

                rows.append({
                    "stratum":    stratum,
                    "bin":        bin_val,
                    "bad_in":     a,
                    "good_in":    b,
                    "bad_out":    c,
                    "good_out":   d,
                    "total":      a + b + c + d,
                    "min_cell":   min_cell,
                    "has_zero":   has_zero,
                    "sparse":     min_cell < self.min_cell_threshold,
                    "odds_ratio": round(or_val, 4) if not np.isnan(or_val) else np.nan,
                })

        result_df = pd.DataFrame(rows)
        self._print(result_df, deal_var, customer_var)
        return result_df

    def _print(
        self,
        result_df: pd.DataFrame,
        deal_var: str,
        customer_var: str,
    ) -> None:
        n_sparse = int(result_df["sparse"].sum())
        n_zero   = int(result_df["has_zero"].sum())

        print(f"\n{'=' * 65}")
        print(f"STRATUM DIAGNOSTICS: {deal_var} stratified by {customer_var}")
        print(f"{'=' * 65}")
        print(result_df.to_string(index=False))
        print(f"\n  Sparse cells (< {self.min_cell_threshold}): {n_sparse}")
        print(f"  Zero cells:                     {n_zero}")

        if n_zero > 0:
            print(
                "  ⚠️  Zero cells will cause undefined odds ratios and "
                "inflate Breslow-Day. Consider merging bins or strata."
            )
        if n_sparse > 0:
            print(
                "  ⚠️  Sparse cells make OR estimates unreliable. "
                "Consider the Tarone correction or merging small bins."
            )


# --------------------------------------------------------------------------
# Deal Variable Diagnostics (multi-variable triage)
# --------------------------------------------------------------------------

class DealVariableDiagnostics:
    """
    Runs StratumDiagnostics across all deal variables and produces a
    one-row-per-variable triage summary.

    Use this immediately after a set of Breslow-Day failures to quickly
    identify which variables are most problematic and why:

        - High sparse_cells / pct_sparse → likely a sample size issue
          → try Tarone correction or merge bins before concluding interaction
        - High zero_cells               → BD result unreliable
          → merge bins or strata first, then retest
        - High or_range with low sparse → genuine interaction
          → investigate the specific stratum driving the divergence

    The summary is sorted by or_range descending so the variables with
    the most inconsistent odds ratios appear at the top.
    """

    def __init__(self, min_cell_threshold: int = ContingencyTable.SPARSE_THRESHOLD):
        self.min_cell_threshold = min_cell_threshold
        self._stratum_diag      = StratumDiagnostics(min_cell_threshold)
        self.detail_results:    Dict[str, pd.DataFrame] = {}

    def run(
        self,
        df: pd.DataFrame,
        deal_vars: List[str],
        customer_var: str,
        outcome_var: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run diagnostics across all deal variables.

        Parameters
        ----------
        df           : DataFrame containing all variables
        deal_vars    : list of binned deal variable column names
        customer_var : stratifying customer variable column
        outcome_var  : binary target column (1 = bad, 0 = good)
        verbose      : if True, prints the full stratum table per variable.
                       Set False to only see the summary.

        Returns
        -------
        Summary DataFrame with one row per deal variable, sorted by
        or_range descending. Full per-stratum detail is stored in
        self.detail_results[variable_name].
        """
        summary_rows = []

        for var in deal_vars:
            if verbose:
                diag = self._stratum_diag.run(df, var, customer_var, outcome_var)
            else:
                # Run silently — suppress the per-variable print
                import io, sys
                _buffer = io.StringIO()
                _old_stdout, sys.stdout = sys.stdout, _buffer
                try:
                    diag = self._stratum_diag.run(df, var, customer_var, outcome_var)
                finally:
                    sys.stdout = _old_stdout

            self.detail_results[var] = diag

            or_vals = diag["odds_ratio"].dropna()
            summary_rows.append({
                "variable":    var,
                "total_cells": len(diag),
                "sparse_cells": int(diag["sparse"].sum()),
                "zero_cells":  int(diag["has_zero"].sum()),
                "pct_sparse":  round(diag["sparse"].mean() * 100, 1),
                "or_min":      round(or_vals.min(), 4) if not or_vals.empty else np.nan,
                "or_max":      round(or_vals.max(), 4) if not or_vals.empty else np.nan,
                "or_range":    round(or_vals.max() - or_vals.min(), 4)
                               if not or_vals.empty else np.nan,
            })

        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values("or_range", ascending=False)
            .reset_index(drop=True)
        )

        self._print_summary(summary_df)
        return summary_df

    def get_detail(self, variable: str) -> pd.DataFrame:
        """Return the full stratum × bin table for a specific variable."""
        if variable not in self.detail_results:
            raise KeyError(
                f"No diagnostic results for '{variable}'. "
                "Call run() first."
            )
        return self.detail_results[variable]

    @staticmethod
    def _print_summary(summary_df: pd.DataFrame) -> None:
        print("\n\n" + "=" * 65)
        print("DEAL VARIABLE DIAGNOSTICS — SUMMARY")
        print("Sorted by OR range descending (widest = most inconsistent)")
        print("=" * 65)
        print(summary_df.to_string(index=False))
        print(
            "\nInterpretation guide:"
            "\n  High or_range, low sparse_cells → likely genuine interaction"
            "\n  High sparse_cells / pct_sparse  → sample size issue; "
            "try Tarone or merge bins"
            "\n  Zero cells present              → BD result unreliable; "
            "merge before retesting"
        )