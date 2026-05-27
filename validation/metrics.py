"""
validation/metrics.py

Model validation covering all three dimensions:

    1. DISCRIMINATION — can the model rank goods from bads?
       Gini coefficient (= 2×AUC - 1)
       KS statistic (max separation of score distributions)

    2. CALIBRATION — are predicted PDs accurate in absolute terms?
       Hosmer-Lemeshow test
       Observed vs predicted default rates by score band

    3. STABILITY — does the model hold up over time?
       PSI — Population Stability Index (overall score distribution)
       CSI — Characteristic Stability Index (per-variable distribution)

Classes:
    DiscriminationMetrics  — Gini and KS
    CalibrationMetrics     — Hosmer-Lemeshow and obs vs pred
    StabilityMetrics       — PSI and CSI
    ValidationReport       — combines all three into a single report
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# 1. Discrimination
# --------------------------------------------------------------------------

class DiscriminationMetrics:
    """
    Gini coefficient and KS statistic.

    Theory:
        Gini = 2 × AUC - 1
        KS   = max|F_goods(score) - F_bads(score)|

    Typical range for retail credit scorecards: Gini 0.45 – 0.65.
    Development vs OOT Gini drop should be ≤ 5 points (0.05).
    """

    GINI_THRESHOLDS = [
        (0.20, "Poor"),
        (0.40, "Acceptable"),
        (0.60, "Good"),
        (0.75, "Strong"),
    ]

    def __init__(
        self,
        y_true: pd.Series,
        y_pred_proba: pd.Series,
        label: str = "",
    ):
        self.y_true      = np.array(y_true)
        self.y_pred_proba = np.array(y_pred_proba)
        self.label       = label

        if len(np.unique(self.y_true)) != 2:
            raise ValueError("y_true must be binary.")

    @property
    def auc(self) -> float:
        return float(roc_auc_score(self.y_true, self.y_pred_proba))

    @property
    def gini(self) -> float:
        return 2 * self.auc - 1

    @property
    def gini_rating(self) -> str:
        for threshold, label in self.GINI_THRESHOLDS:
            if self.gini < threshold:
                return label
        return "Very Strong — check for data leakage"

    @property
    def ks(self) -> Tuple[float, float]:
        """
        Returns (KS statistic, score threshold at which KS is maximised).
        The threshold is a useful starting point for cut-off decisions.
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        ks_vals = tpr - fpr
        idx     = int(np.argmax(ks_vals))
        return float(ks_vals[idx]), float(thresholds[idx])

    def summary(self) -> Dict:
        ks_stat, ks_threshold = self.ks
        return {
            "label":         self.label,
            "auc":           round(self.auc, 4),
            "gini":          round(self.gini, 4),
            "gini_rating":   self.gini_rating,
            "ks_statistic":  round(ks_stat, 4),
            "ks_threshold":  round(ks_threshold, 4),
        }


# --------------------------------------------------------------------------
# 2. Calibration
# --------------------------------------------------------------------------

class CalibrationMetrics:
    """
    Hosmer-Lemeshow goodness-of-fit test and observed vs predicted comparison.

    Hosmer-Lemeshow:
        H0: model is well calibrated (no significant difference between
            predicted PD and observed default rate)
        WANT p > 0.05 (fail to reject H0)
        p < 0.05 → recalibration of intercept may be needed
    """

    def __init__(
        self,
        y_true: pd.Series,
        y_pred_proba: pd.Series,
        n_groups: int = 10,
    ):
        self.y_true       = np.array(y_true)
        self.y_pred_proba = np.array(y_pred_proba)
        self.n_groups     = n_groups

    def hosmer_lemeshow(self) -> Dict:
        df = pd.DataFrame({"y": self.y_true, "p": self.y_pred_proba})
        df["group"] = pd.qcut(
            df["p"], q=self.n_groups, duplicates="drop", labels=False
        )

        grouped = (
            df.groupby("group")
            .agg(n=("y", "count"), observed=("y", "sum"), expected=("p", "sum"), mean_p=("p", "mean"))
            .reset_index()
        )

        # HL statistic
        hl_stat = 0.0
        for _, row in grouped.iterrows():
            p_bar = row["mean_p"]
            if 0 < p_bar < 1:
                hl_stat += (row["observed"] - row["expected"]) ** 2 / (
                    row["n"] * p_bar * (1 - p_bar)
                )

        dof     = self.n_groups - 2
        p_value = 1 - stats.chi2.cdf(hl_stat, df=dof)
        calibrated = p_value > 0.05

        return {
            "statistic":     round(hl_stat, 4),
            "p_value":       round(p_value, 4),
            "degrees_of_freedom": dof,
            "well_calibrated": calibrated,
            "group_stats":   grouped,
            "recommendation": (
                "✅  Model is well calibrated."
                if calibrated else
                f"⚠️  Miscalibration detected (p={p_value:.4f}). "
                "Consider adjusting the model intercept to match observed rates."
            ),
        }

    def observed_vs_predicted(
        self,
        score_bands: Optional[pd.Series] = None,
        n_bands: int = 10,
    ) -> pd.DataFrame:
        """
        Group customers into bands and compare predicted PD to observed
        default rate.

        score_bands : optional pre-computed score column for banding.
                      If None, bands are created from predicted probability.
        """
        df = pd.DataFrame({"y": self.y_true, "p": self.y_pred_proba})
        band_col = score_bands if score_bands is not None else df["p"]
        df["band"] = pd.qcut(band_col, q=n_bands, duplicates="drop")

        result = (
            df.groupby("band", observed=True)
            .agg(n_total=("y", "count"), n_bads=("y", "sum"), predicted_pd=("p", "mean"))
            .reset_index()
        )

        result["observed_bad_rate"] = result["n_bads"] / result["n_total"]
        result["difference"]        = (
            result["observed_bad_rate"] - result["predicted_pd"]
        )
        result["pct_difference"]    = (
            result["difference"] / result["predicted_pd"] * 100
        ).round(1)

        return result


# --------------------------------------------------------------------------
# 3. Stability
# --------------------------------------------------------------------------

class StabilityMetrics:
    """
    PSI and CSI for monitoring distribution shifts over time.

    PSI thresholds:
        < 0.10  : Stable         — monitor normally
        0.10-0.25: Moderate shift — investigate
        > 0.25  : Significant    — model likely needs rebuilding

    CSI uses the same formula per variable to diagnose which input
    is driving an overall PSI shift.
    """

    PSI_STABLE   = 0.10
    PSI_MODERATE = 0.25

    @staticmethod
    def _psi_from_distributions(
        expected_pct: pd.Series,
        actual_pct: pd.Series,
    ) -> Tuple[float, pd.DataFrame]:
        """Core PSI calculation from two percentage distributions."""
        # Protect against zeros
        expected_pct = expected_pct.clip(lower=1e-4)
        actual_pct   = actual_pct.clip(lower=1e-4)

        psi_contribs = (actual_pct - expected_pct) * np.log(
            actual_pct / expected_pct
        )
        psi = psi_contribs.sum()

        detail = pd.DataFrame({
            "bin":             expected_pct.index,
            "expected_pct":    expected_pct.values,
            "actual_pct":      actual_pct.values,
            "psi_contribution": psi_contribs.values,
        })
        return float(psi), detail

    @classmethod
    def calculate_psi(
        cls,
        expected: pd.Series,
        actual: pd.Series,
        n_bins: int = 10,
        label: str = "score",
    ) -> Dict:
        """
        Calculate PSI between development (expected) and
        monitoring (actual) score distributions.
        """
        # Define bins from expected distribution
        _, bin_edges = pd.qcut(
            expected, q=n_bins, duplicates="drop", retbins=True
        )
        bin_edges[0]  = -np.inf
        bin_edges[-1] = np.inf

        expected_counts = (
            pd.cut(expected, bins=bin_edges)
            .value_counts(normalize=True)
            .sort_index()
        )
        actual_counts = (
            pd.cut(actual, bins=bin_edges)
            .value_counts(normalize=True)
            .sort_index()
        )

        # Align to same bins
        all_bins      = expected_counts.index.union(actual_counts.index)
        expected_pct  = expected_counts.reindex(all_bins, fill_value=1e-4)
        actual_pct    = actual_counts.reindex(all_bins,   fill_value=1e-4)

        psi, detail = cls._psi_from_distributions(expected_pct, actual_pct)
        status, action = cls._psi_interpretation(psi)

        return {
            "label":      label,
            "psi":        round(psi, 4),
            "status":     status,
            "action":     action,
            "bin_detail": detail,
        }

    @classmethod
    def calculate_csi(
        cls,
        variable: str,
        expected: pd.Series,
        actual: pd.Series,
        n_bins: int = 10,
    ) -> Dict:
        """CSI for a single variable — same formula as PSI."""
        result = cls.calculate_psi(expected, actual, n_bins, label=variable)
        result["metric"] = "CSI"
        return result

    @classmethod
    def run_csi_all(
        cls,
        expected_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        variables: List[str],
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Run CSI for all variables and return a summary DataFrame,
        sorted by CSI descending so the most shifted variables
        appear at the top.
        """
        rows = []
        for var in variables:
            if var in expected_df.columns and var in actual_df.columns:
                res = cls.calculate_csi(
                    var, expected_df[var], actual_df[var], n_bins
                )
                rows.append({
                    "variable": var,
                    "csi":      res["psi"],
                    "status":   res["status"],
                    "action":   res["action"],
                })
        return (
            pd.DataFrame(rows)
            .sort_values("csi", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def _psi_interpretation(psi: float) -> Tuple[str, str]:
        if psi < StabilityMetrics.PSI_STABLE:
            return "Stable", "Monitor normally."
        elif psi < StabilityMetrics.PSI_MODERATE:
            return (
                "Moderate Shift",
                "Investigate driving variables via CSI. Increase monitoring frequency.",
            )
        return (
            "Significant Shift",
            "Model likely needs rebuilding. Review all variable distributions.",
        )


# --------------------------------------------------------------------------
# Validation Report
# --------------------------------------------------------------------------

class ValidationReport:
    """
    Runs and presents a complete model validation report covering
    discrimination, calibration, and stability.

    Supports an optional validation set (val) sitting between development
    and OOT. When present, the discrimination section reports all three
    samples side-by-side and flags Gini drops across the full chain:

        dev → val  : should be modest (same period, held-out slice)
        val → OOT  : key temporal stability check
        dev → OOT  : headline governance figure
    """

    GINI_DROP_THRESHOLD = 0.05   # 5 Gini points — standard governance threshold

    def __init__(self, model_name: str = "Scorecard Model"):
        self.model_name = model_name
        self.results: Dict = {}

    def run(
        self,
        # Development sample (required)
        y_true_dev:   pd.Series,
        y_pred_dev:   pd.Series,
        scores_dev:   pd.Series,
        # OOT sample (required)
        y_true_oot:   pd.Series,
        y_pred_oot:   pd.Series,
        scores_oot:   pd.Series,
        # Variable-level stability (required)
        vars_dev:     pd.DataFrame,
        vars_oot:     pd.DataFrame,
        variables:    List[str],
        # Validation set (optional)
        y_true_val:   Optional[pd.Series]    = None,
        y_pred_val:   Optional[pd.Series]    = None,
        scores_val:   Optional[pd.Series]    = None,
        vars_val:     Optional[pd.DataFrame] = None,
    ) -> "ValidationReport":
        """
        Run the full validation suite.

        Parameters
        ----------
        y_true_dev/val/oot  : observed default flags
        y_pred_dev/val/oot  : model-predicted PD probabilities
        scores_dev/val/oot  : integer scorecard scores
        vars_dev/val/oot    : DataFrames of WoE variable values for CSI
        variables           : variable names to run CSI on
        y_true/pred/scores/vars_val : optional validation set — pass all
                                      four or none (partial is not supported)
        """
        has_val = y_true_val is not None

        if has_val and any(
            x is None for x in [y_pred_val, scores_val, vars_val]
        ):
            raise ValueError(
                "Validation set is partially specified. "
                "Provide y_true_val, y_pred_val, scores_val, and vars_val "
                "together, or omit all four."
            )

        # ------------------------------------------------------------------
        # 1. Discrimination
        # ------------------------------------------------------------------
        disc_dev = DiscriminationMetrics(y_true_dev, y_pred_dev, label="Development")
        disc_oot = DiscriminationMetrics(y_true_oot, y_pred_oot, label="OOT")

        discrimination: Dict = {
            "development": disc_dev.summary(),
            "oot":         disc_oot.summary(),
            "validation":  None,
            # dev → OOT headline drop
            "gini_drop_dev_oot":      round(disc_dev.gini - disc_oot.gini, 4),
            "gini_drop_dev_oot_flag": self._drop_flag(disc_dev.gini - disc_oot.gini),
            # val chain drops — populated below if val is present
            "gini_drop_dev_val":      None,
            "gini_drop_val_oot":      None,
        }

        if has_val:
            disc_val = DiscriminationMetrics(y_true_val, y_pred_val, label="Validation")
            discrimination["validation"]       = disc_val.summary()
            discrimination["gini_drop_dev_val"] = round(disc_dev.gini - disc_val.gini, 4)
            discrimination["gini_drop_val_oot"] = round(disc_val.gini - disc_oot.gini, 4)

        # ------------------------------------------------------------------
        # 2. Calibration — development sample only.
        #    Validation/OOT may not have fully matured outcomes yet.
        # ------------------------------------------------------------------
        calib    = CalibrationMetrics(y_true_dev, y_pred_dev)
        hl       = calib.hosmer_lemeshow()
        obs_pred = calib.observed_vs_predicted(scores_dev)

        # ------------------------------------------------------------------
        # 3. Stability
        #    PSI: dev → OOT (primary), dev → val (secondary when val present)
        #    CSI: dev → OOT (always), dev → val (when val present)
        # ------------------------------------------------------------------
        psi_dev_oot = StabilityMetrics.calculate_psi(
            scores_dev, scores_oot, label="Dev → OOT Score PSI"
        )
        csi_dev_oot = StabilityMetrics.run_csi_all(vars_dev, vars_oot, variables)

        stability: Dict = {
            "psi_dev_oot": psi_dev_oot,
            "csi_dev_oot": csi_dev_oot,
            "psi_dev_val": None,
            "csi_dev_val": None,
        }

        if has_val:
            stability["psi_dev_val"] = StabilityMetrics.calculate_psi(
                scores_dev, scores_val, label="Dev → Val Score PSI"
            )
            stability["csi_dev_val"] = StabilityMetrics.run_csi_all(
                vars_dev, vars_val, variables
            )

        self.results = {
            "has_validation": has_val,
            "discrimination": discrimination,
            "calibration":    {"hosmer_lemeshow": hl, "observed_vs_predicted": obs_pred},
            "stability":      stability,
        }
        return self

    # ------------------------------------------------------------------
    # Report printing
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        if not self.results:
            print("No results — call run() first.")
            return

        has_val = self.results["has_validation"]
        disc    = self.results["discrimination"]
        calib   = self.results["calibration"]
        stab    = self.results["stability"]
        hl      = calib["hosmer_lemeshow"]

        print("\n" + "=" * 72)
        print(f"  VALIDATION REPORT — {self.model_name}")
        print("=" * 72)

        # ------------------------------------------------------------------
        # 1. Discrimination
        # ------------------------------------------------------------------
        print("\n  1. DISCRIMINATION")
        print("  " + "-" * 60)

        dev = disc["development"]
        oot = disc["oot"]
        val = disc["validation"]

        if has_val:
            print(
                f"  {'Metric':<25} {'Development':>14} "
                f"{'Validation':>12} {'OOT':>12}"
            )
            print(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*12}")
            print(
                f"  {'Gini':<25} {dev['gini']:>14.4f} "
                f"{val['gini']:>12.4f} {oot['gini']:>12.4f}"
            )
            print(
                f"  {'KS Statistic':<25} {dev['ks_statistic']:>14.4f} "
                f"{val['ks_statistic']:>12.4f} {oot['ks_statistic']:>12.4f}"
            )
            print(
                f"  {'AUC':<25} {dev['auc']:>14.4f} "
                f"{val['auc']:>12.4f} {oot['auc']:>12.4f}"
            )

            drop_dv = disc["gini_drop_dev_val"]
            drop_vo = disc["gini_drop_val_oot"]
            drop_do = disc["gini_drop_dev_oot"]

            print(f"\n  Gini drop  Dev → Val : {drop_dv:+.4f}  {self._drop_flag(drop_dv)}")
            print(f"  Gini drop  Val → OOT : {drop_vo:+.4f}  {self._drop_flag(drop_vo)}")
            print(f"  Gini drop  Dev → OOT : {drop_do:+.4f}  {self._drop_flag(drop_do)}  (governance headline)")
        else:
            print(f"  {'Metric':<25} {'Development':>14} {'OOT':>12}")
            print(f"  {'-'*25} {'-'*14} {'-'*12}")
            print(
                f"  {'Gini':<25} {dev['gini']:>14.4f} {oot['gini']:>12.4f}  "
                f"{disc['gini_drop_dev_oot_flag']}  drop={disc['gini_drop_dev_oot']:.4f}"
            )
            print(f"  {'KS Statistic':<25} {dev['ks_statistic']:>14.4f} {oot['ks_statistic']:>12.4f}")
            print(f"  {'AUC':<25} {dev['auc']:>14.4f} {oot['auc']:>12.4f}")

        print(f"\n  Gini Rating (Dev): {dev['gini_rating']}")

        if disc["gini_drop_dev_oot"] > self.GINI_DROP_THRESHOLD:
            print(
                f"  ⚠️  Dev → OOT Gini drop of {disc['gini_drop_dev_oot']:.4f} "
                "exceeds 5 points — review for overfitting or population shift."
            )

        # ------------------------------------------------------------------
        # 2. Calibration
        # ------------------------------------------------------------------
        print("\n  2. CALIBRATION  (Development sample)")
        print("  " + "-" * 60)
        hl_flag = "✅" if hl["well_calibrated"] else "⚠️"
        print(
            f"  Hosmer-Lemeshow:  χ²={hl['statistic']:.4f}, "
            f"p={hl['p_value']:.4f}  {hl_flag}"
        )
        print(f"  {hl['recommendation']}")
        print("\n  Observed vs Predicted (Development, by score band):")
        obs_p = calib["observed_vs_predicted"][[
            "band", "n_total", "observed_bad_rate", "predicted_pd", "pct_difference"
        ]].rename(columns={
            "n_total":           "N",
            "observed_bad_rate": "Obs. Rate",
            "predicted_pd":      "Pred. PD",
            "pct_difference":    "Diff %",
        })
        print(obs_p.to_string(index=False))

        # ------------------------------------------------------------------
        # 3. Stability
        # ------------------------------------------------------------------
        print("\n  3. STABILITY")
        print("  " + "-" * 60)

        psi_do      = stab["psi_dev_oot"]
        psi_do_flag = "✅" if psi_do["status"] == "Stable" else "⚠️"
        print(f"  PSI (Dev → OOT): {psi_do['psi']:.4f} — {psi_do['status']}  {psi_do_flag}")
        print(f"  Action: {psi_do['action']}")

        if has_val and stab["psi_dev_val"] is not None:
            psi_dv      = stab["psi_dev_val"]
            psi_dv_flag = "✅" if psi_dv["status"] == "Stable" else "⚠️"
            print(f"\n  PSI (Dev → Val): {psi_dv['psi']:.4f} — {psi_dv['status']}  {psi_dv_flag}")
            print(f"  Action: {psi_dv['action']}")

        print("\n  CSI by Variable  (Dev → OOT):")
        print(f"  {'Variable':<30} {'CSI':>6}  {'Status':<20} Action")
        print("  " + "-" * 68)
        for _, row in stab["csi_dev_oot"].iterrows():
            flag = "✅" if row["status"] == "Stable" else "⚠️"
            print(
                f"  {row['variable']:<30} {row['csi']:>6.4f}  "
                f"{row['status']:<20} {flag} {row['action']}"
            )

        if has_val and stab["csi_dev_val"] is not None:
            print("\n  CSI by Variable  (Dev → Val):")
            print(f"  {'Variable':<30} {'CSI':>6}  {'Status':<20} Action")
            print("  " + "-" * 68)
            for _, row in stab["csi_dev_val"].iterrows():
                flag = "✅" if row["status"] == "Stable" else "⚠️"
                print(
                    f"  {row['variable']:<30} {row['csi']:>6.4f}  "
                    f"{row['status']:<20} {flag} {row['action']}"
                )

        print("\n" + "=" * 72)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_flag(drop: float) -> str:
        return "✅" if drop <= ValidationReport.GINI_DROP_THRESHOLD else "⚠️"

    def discrimination_comparison(self) -> pd.DataFrame:
        """
        Return discrimination metrics as a DataFrame.
        Includes validation row when a validation set was provided.
        """
        if not self.results:
            raise RuntimeError("Call run() first.")
        disc = self.results["discrimination"]
        rows = [disc["development"]]
        if disc["validation"] is not None:
            rows.append(disc["validation"])
        rows.append(disc["oot"])
        return pd.DataFrame(rows).set_index("label")