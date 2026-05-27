"""
modelling/model_comparison.py

Comparison framework for multiple InteractionLogisticRegression models
fitted with different deal variable combinations.

Covers four comparison dimensions:

    1. Model fit      — AIC, BIC, log-likelihood ranking
    2. LR tests       — likelihood ratio tests for nested model pairs
    3. Discrimination — Gini, KS, AUC on development and OOT samples
    4. Interpretability — coefficient table and per-bin point contributions
                          at multiple Equifax score percentiles

Point contribution note:
    With interaction terms, the scorecard points for a deal variable bin
    depend on the customer's Equifax score. A bin is no longer worth a
    single fixed number of points — it is worth a range of points that
    varies with credit quality.

    This pipeline computes that range at P25/P50/P75 of the standardised
    Equifax score, giving analysts a clear picture of how the model
    rewards or penalises deal risk differently across the customer base.

Likelihood Ratio Test recap:
    LR = 2 × (LL_full − LL_restricted) ~ χ²(df_full − df_restricted)
    H0 : restricted model fits as well as the full model
    p < 0.05 → full model is significantly better (extra terms justified)
    Only valid for nested models (restricted vars ⊂ full vars).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Optional, Tuple

from .interaction_model import InteractionLogisticRegression


class ModelComparison:
    """
    Compares multiple fitted InteractionLogisticRegression models across
    statistical fit, discrimination, and scorecard interpretability.

    Usage
    -----
    comparison = ModelComparison(models)
    comparison.print_comparison(df_dev, y_dev, df_oot, y_oot)
    best = comparison.best_model(criterion="aic")
    contrib = comparison.point_contribution_breakdown(df_dev, bin_stats)
    """

    def __init__(self, models: List[InteractionLogisticRegression]) -> None:
        if len(models) < 2:
            raise ValueError(
                "At least two fitted models are required for comparison."
            )
        self.models: List[InteractionLogisticRegression] = models

    # ------------------------------------------------------------------
    # 1. AIC / BIC
    # ------------------------------------------------------------------

    def aic_bic_table(self) -> pd.DataFrame:
        """
        AIC and BIC for all models, ranked by AIC ascending.

        Lower AIC/BIC indicates a better balance of fit and complexity.
        Δ AIC > 10 vs the best model is a strong reason to prefer the
        lower-AIC model. Δ AIC < 2 means models are indistinguishable
        by this criterion — prefer simpler in that case.

        Columns: model | deal_variables | n_params | log_likelihood |
                 aic | bic | aic_rank | delta_aic
        """
        rows = []
        for m in self.models:
            rows.append({
                "model":          m.model_name,
                "deal_variables": ", ".join(m.deal_variables),
                "n_params":       m.diagnostics["n_params"],
                "log_likelihood": round(m.diagnostics["log_likelihood"], 4),
                "aic":            round(m.diagnostics["aic"], 2),
                "bic":            round(m.diagnostics["bic"], 2),
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("aic")
            .reset_index(drop=True)
        )
        df["aic_rank"]  = range(1, len(df) + 1)
        df["delta_aic"] = (df["aic"] - df["aic"].min()).round(2)
        return df

    # ------------------------------------------------------------------
    # 2. Likelihood ratio tests
    # ------------------------------------------------------------------

    def likelihood_ratio_test(
        self,
        restricted_model: InteractionLogisticRegression,
        full_model:       InteractionLogisticRegression,
    ) -> Dict:
        """
        Test whether the additional terms in full_model significantly
        improve fit over restricted_model.

        LR = 2 × (LL_full − LL_restricted) ~ χ²(df_full − df_restricted)

        Parameters
        ----------
        restricted_model : model with fewer parameters (must be nested in full)
        full_model       : model with more parameters

        Returns
        -------
        Dict with: restricted_model, full_model, lr_statistic, df_difference,
                   p_value, significant, recommendation
        """
        ll_r  = restricted_model.diagnostics["log_likelihood"]
        ll_f  = full_model.diagnostics["log_likelihood"]
        df_r  = restricted_model.diagnostics["n_params"]
        df_f  = full_model.diagnostics["n_params"]

        if df_f <= df_r:
            raise ValueError(
                f"full_model must have more parameters than restricted_model. "
                f"full={df_f} params, restricted={df_r} params."
            )

        lr_stat = 2 * (ll_f - ll_r)
        df_diff = df_f - df_r
        p_value = 1 - stats.chi2.cdf(lr_stat, df=df_diff)

        significant = bool(p_value < 0.05)
        recommendation = (
            f"Full model ({full_model.model_name}) is significantly better "
            f"(p={p_value:.4f}) — additional interaction terms are justified."
            if significant else
            f"No significant improvement over restricted model (p={p_value:.4f}). "
            f"Prefer simpler model ({restricted_model.model_name}) on parsimony grounds."
        )

        return {
            "restricted_model": restricted_model.model_name,
            "full_model":       full_model.model_name,
            "lr_statistic":     round(lr_stat, 4),
            "df_difference":    int(df_diff),
            "p_value":          round(p_value, 4),
            "significant":      significant,
            "recommendation":   recommendation,
        }

    def pairwise_lr_tests(self) -> pd.DataFrame:
        """
        Run LR tests for all genuinely nested design matrices.

        Nesting is determined from fitted design terms, not just deal variable
        names. This captures the key governance comparison where a main-only
        model and a full interaction model use the same deal variables, but the
        full model adds Equifax x deal interaction terms.

        Non-nested pairs are skipped — LR tests are not valid for them.

        Columns: restricted | full | added_terms | lr_stat | df_diff |
                 p_value | significant
        """
        rows = []
        for m_r in self.models:
            for m_f in self.models:
                if m_r is m_f:
                    continue

                r_terms = set(getattr(m_r, "_design_columns", []))
                f_terms = set(getattr(m_f, "_design_columns", []))

                if r_terms < f_terms:  # strict subset -> nested
                    result = self.likelihood_ratio_test(m_r, m_f)
                    added_terms = sorted(f_terms - r_terms)
                    rows.append({
                        "restricted":  result["restricted_model"],
                        "full":        result["full_model"],
                        "added_terms": ", ".join(added_terms),
                        "lr_stat":     result["lr_statistic"],
                        "df_diff":     result["df_difference"],
                        "p_value":     result["p_value"],
                        "significant": result["significant"],
                    })

        if not rows:
            return pd.DataFrame(
                columns=["restricted", "full", "added_terms", "lr_stat",
                         "df_diff", "p_value", "significant"]
            )

        return (
            pd.DataFrame(rows)
            .sort_values("p_value")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # 3. Discrimination
    # ------------------------------------------------------------------

    @staticmethod
    def _gini_ks(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[float, float]:
        auc  = float(roc_auc_score(y_true, y_pred))
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks   = float(np.max(tpr - fpr))
        return round(gini, 4), round(ks, 4)

    def discrimination_table(
        self,
        df_dev:     pd.DataFrame,
        y_true_dev: pd.Series,
        df_oot:     Optional[pd.DataFrame] = None,
        y_true_oot: Optional[pd.Series]    = None,
    ) -> pd.DataFrame:
        """
        Gini, KS, and AUC for all models on development (and optionally OOT).

        Sorted by development Gini descending.
        Gini drop (dev − OOT) flagged if > 5 points (0.05).

        Columns: model | deal_variables | gini_dev | ks_dev
                 [| gini_oot | ks_oot | gini_drop | gini_drop_flag]
        """
        rows = []
        for m in self.models:
            y_pred_dev = m.predict_proba(df_dev)
            gini_dev, ks_dev = self._gini_ks(y_true_dev.values, y_pred_dev)

            row: Dict = {
                "model":          m.model_name,
                "deal_variables": ", ".join(m.deal_variables),
                "gini_dev":       gini_dev,
                "ks_dev":         ks_dev,
            }

            if df_oot is not None and y_true_oot is not None:
                y_pred_oot = m.predict_proba(df_oot)
                gini_oot, ks_oot = self._gini_ks(y_true_oot.values, y_pred_oot)
                drop = round(gini_dev - gini_oot, 4)
                row.update({
                    "gini_oot":       gini_oot,
                    "ks_oot":         ks_oot,
                    "gini_drop":      drop,
                    "gini_drop_flag": "✅" if drop <= 0.05 else "⚠️",
                })

            rows.append(row)

        return (
            pd.DataFrame(rows)
            .sort_values("gini_dev", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # 4a. Coefficient comparison
    # ------------------------------------------------------------------

    def coefficient_comparison(self) -> pd.DataFrame:
        """
        Side-by-side coefficient table across all models.

        Each cell shows: "{coefficient}* (p={p_value})" where * marks
        significance at 5%. Missing terms (variable not in that model)
        are shown as "—".

        Columns: term | term_type | {model_name_1} | {model_name_2} | ...
        """
        all_terms: List[str] = []
        seen: set = set()
        for m in self.models:
            for t in m.coefficients.index:
                if t not in seen:
                    all_terms.append(t)
                    seen.add(t)

        data: Dict = {
            "term":      all_terms,
            "term_type": [self._classify_term(t) for t in all_terms],
        }

        for m in self.models:
            col_values = []
            for term in all_terms:
                if term in m.coefficients.index:
                    coef = m.coefficients[term]
                    pval = m.diagnostics["p_values"][term]
                    sig  = "*" if pval < 0.05 else ""
                    col_values.append(f"{coef:.4f}{sig} (p={pval:.3f})")
                else:
                    col_values.append("—")
            data[m.model_name] = col_values

        return pd.DataFrame(data)

    @staticmethod
    def _classify_term(term: str) -> str:
        if term.startswith("equifax_x_"):
            return "Interaction"
        elif term == "equifax_std":
            return "Equifax (Main)"
        elif term.startswith("cont_"):
            return "Deal Continuous (Main)"
        elif term.startswith("woe_"):
            return "Deal WoE (Main)"
        else:
            return "Deal Main"

    # ------------------------------------------------------------------
    # 4b. Point contribution breakdown
    # ------------------------------------------------------------------

    def point_contribution_breakdown(
        self,
        df:        pd.DataFrame,
        bin_stats: Dict[str, pd.DataFrame],
        pdo:        float = 20,
        base_score: float = 600,
        base_odds:  float = 50,
    ) -> pd.DataFrame:
        """
        Scorecard point contribution per WoE bin at P25/P50/P75 Equifax.

        Because interaction terms make a deal variable's point contribution
        dependent on the customer's Equifax score, a single fixed points
        value per bin is inappropriate. This method computes:

            Points_j(bin_b, Equifax_p) =
                −(β_j + β_ij × Equifax_std_p) × WoE_b × B

        at the 25th, 50th, and 75th percentile of the standardised Equifax
        score in df, for each bin of each deal variable.

        Interpretation:
            A row showing Points = +12 at P25 and +6 at P75 means the
            high-WoE bin adds 12 risk points for a weaker-credit customer
            but only 6 for a stronger-credit customer — the deal variable
            is more discriminating at lower credit quality.

        Parameters
        ----------
        df        : development dataframe (used to compute Equifax percentiles)
        bin_stats : {variable_name: bin_stats_DataFrame}
                    from BinningPipeline.get_all_bin_stats()
        pdo, base_score, base_odds : standard scorecard scaling parameters

        Returns
        -------
        DataFrame: model | variable | bin | woe | bad_rate | n_total |
                   term_type | equifax_percentile | equifax_std | points
        """
        B: float = pdo / np.log(2)
        rows = []

        for m in self.models:
            # Compute Equifax percentiles in standardised space
            eq_std_vals: np.ndarray = m.scaler.transform(
                df[[m.equifax_col]]
            ).flatten()

            percentiles: Dict[str, float] = {
                "P25": float(np.percentile(eq_std_vals, 25)),
                "P50": float(np.percentile(eq_std_vals, 50)),
                "P75": float(np.percentile(eq_std_vals, 75)),
            }

            n_terms        = len(m._design_columns)
            intercept_share = m.intercept / n_terms

            # --- Equifax main effect (not bin-specific) ---
            beta_eq = float(m.coefficients["equifax_std"])
            for pct_label, eq_val in percentiles.items():
                # Points per 1-std increase in Equifax at this percentile
                points = -(beta_eq + intercept_share) * B
                rows.append({
                    "model":              m.model_name,
                    "variable":           m.equifax_col,
                    "bin":                "per std unit",
                    "woe":                None,
                    "bad_rate":           None,
                    "n_total":            None,
                    "term_type":          "Equifax (Main)",
                    "equifax_percentile": pct_label,
                    "equifax_std":        round(eq_val, 3),
                    "points":             round(points, 2),
                })

            # --- Deal variable bins ---
            for var in m.deal_variables:
                if var not in bin_stats:
                    continue

                beta_main = float(m.coefficients.get(f"woe_{var}", 0.0))
                beta_int  = float(m.coefficients.get(f"equifax_x_{var}", 0.0))

                for _, bin_row in bin_stats[var].iterrows():
                    woe_val: float = float(bin_row["woe"])

                    for pct_label, eq_val in percentiles.items():
                        # Effective slope at this Equifax percentile
                        effective_slope = beta_main + beta_int * eq_val
                        # Points contribution of this WoE bin value
                        points = -(effective_slope + intercept_share) * woe_val * B

                        rows.append({
                            "model":              m.model_name,
                            "variable":           var,
                            "bin":                str(bin_row["bin"]),
                            "woe":                round(woe_val, 4),
                            "bad_rate":           round(float(bin_row["bad_rate"]), 4),
                            "n_total":            int(bin_row["n_total"]),
                            "term_type":          "Deal Variable",
                            "equifax_percentile": pct_label,
                            "equifax_std":        round(eq_val, 3),
                            "points":             round(points, 2),
                        })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Best model selection
    # ------------------------------------------------------------------

    def best_model(
        self,
        criterion: str = "aic",
    ) -> InteractionLogisticRegression:
        """
        Return the best-fitting model by AIC or BIC (lower = better).

        Parameters
        ----------
        criterion : 'aic' or 'bic'
        """
        valid = ("aic", "bic")
        if criterion not in valid:
            raise ValueError(f"criterion must be one of {valid}.")
        return min(self.models, key=lambda m: m.diagnostics[criterion])

    # ------------------------------------------------------------------
    # Full comparison report
    # ------------------------------------------------------------------

    def print_comparison(
        self,
        df_dev:     pd.DataFrame,
        y_true_dev: pd.Series,
        df_oot:     Optional[pd.DataFrame] = None,
        y_true_oot: Optional[pd.Series]    = None,
    ) -> None:
        """
        Print the full model comparison report to stdout.

        Sections:
            1. AIC / BIC ranking
            2. Likelihood ratio tests (nested pairs only)
            3. Discrimination metrics (Gini, KS)
            4. Coefficient comparison

        For point contribution breakdown, call point_contribution_breakdown()
        separately (requires bin_stats from BinningPipeline).
        """
        sep = "=" * 72

        print(f"\n{sep}")
        print("  MODEL COMPARISON REPORT")
        print(sep)

        # 1. AIC / BIC
        print("\n  1. AIC / BIC RANKING  (lower = better)")
        print("  " + "-" * 60)
        print(self.aic_bic_table().to_string(index=False))
        print(
            "\n  Δ AIC interpretation: < 2 → indistinguishable;  "
            "2–10 → moderate preference;  > 10 → strong preference"
        )

        # 2. LR tests
        print(f"\n\n  2. LIKELIHOOD RATIO TESTS  (nested pairs only)")
        print("  " + "-" * 60)
        lr_df = self.pairwise_lr_tests()
        if lr_df.empty:
            print(
                "  No nested model pairs found.\n"
                "  LR tests require one model's deal variables to be a strict\n"
                "  subset of another's. Check model combinations."
            )
        else:
            print(lr_df.to_string(index=False))

        # 3. Discrimination
        print(f"\n\n  3. DISCRIMINATION METRICS")
        print("  " + "-" * 60)
        disc = self.discrimination_table(df_dev, y_true_dev, df_oot, y_true_oot)
        print(disc.to_string(index=False))
        if "gini_drop" in disc.columns:
            print(
                "\n  Gini drop flag: ✅ = drop ≤ 0.05 (acceptable);  "
                "⚠️  = drop > 0.05 (review for overfitting)"
            )

        # 4. Coefficients
        print(f"\n\n  4. COEFFICIENT COMPARISON  (* = significant at 5%)")
        print("  " + "-" * 60)
        print(self.coefficient_comparison().to_string(index=False))

        print(f"\n{sep}\n")
