"""
modelling/scorecard_scaler.py

Converts logistic regression coefficients and WoE bin values
into an integer-points scorecard that credit analysts can use directly.

Theory recap:
    The scorecard score is a linear rescaling of the log-odds:
        Score = A - B × log-odds

    Where:
        B = PDO / ln(2)
        A = base_score - B × ln(base_odds)

        PDO        : Points to Double the Odds (typically 20)
        base_score : Score at which base_odds applies (typically 600)
        base_odds  : Odds (goods:bads) at base_score (e.g. 50 means 50:1).
                     Logistic models use bad:good log-odds internally, hence
                     the minus sign in A.

    Per-variable points for bin i of variable j:
        Points_ij = -(βj × WoE_ij + β0/n) × B

    The negative sign ensures higher risk → lower score (convention).

    Final customer score = A + sum of points from each selected bin.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .logistic_model import ScorecardLogisticRegression


class ScorecardScaler:
    """
    Builds a points-based scorecard from a fitted logistic regression
    and the WoE bin statistics from the binning pipeline.

    Usage
    -----
    scaler = ScorecardScaler(pdo=20, base_score=600, base_odds=50)
    scorecard_table = scaler.build(model, bin_stats_dict)
    scores = scaler.score(df_woe, model, bin_stats_dict)
    print(scaler.display())
    """

    def __init__(
        self,
        pdo: float        = 20,
        base_score: float = 600,
        base_odds: float  = 50,
    ):
        """
        Parameters
        ----------
        pdo        : Points to Double the Odds (standard: 20)
        base_score : Score at the base_odds reference point (standard: 600)
        base_odds  : Goods-to-bads ratio at base_score (standard: 50)
        """
        self.pdo        = pdo
        self.base_score = base_score
        self.base_odds  = base_odds

        # Derived scaling constants
        self.B = pdo / np.log(2)
        self.A = base_score - self.B * np.log(base_odds)

        self.scorecard_table: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Build scorecard table
    # ------------------------------------------------------------------

    def build(
        self,
        model: ScorecardLogisticRegression,
        bin_stats: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute the points for each bin of each variable.

        Parameters
        ----------
        model     : fitted ScorecardLogisticRegression
        bin_stats : dict of {variable_name: bin_stats DataFrame}
                    (use BinningPipeline.get_all_bin_stats())

        Returns
        -------
        DataFrame with columns: variable, bin, woe, bad_rate, n_total, points
        """
        n_vars = len(model.variables)
        rows   = []

        for variable in model.variables:
            if variable not in bin_stats:
                raise KeyError(
                    f"No bin stats found for '{variable}'. "
                    "Check BinningPipeline.get_all_bin_stats()."
                )

            beta           = model.coefficients[variable]
            intercept_share = model.intercept / n_vars  # spread evenly across vars

            for _, row in bin_stats[variable].iterrows():
                points = -(beta * row["woe"] + intercept_share) * self.B

                rows.append({
                    "variable": variable,
                    "bin":      str(row["bin"]),
                    "woe":      round(row["woe"], 4),
                    "bad_rate": round(row["bad_rate"], 4),
                    "n_total":  int(row["n_total"]),
                    "points":   int(round(points)),
                })

        self.scorecard_table = pd.DataFrame(rows)
        return self.scorecard_table

    # ------------------------------------------------------------------
    # Score a dataset
    # ------------------------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        model: ScorecardLogisticRegression,
        bin_stats: Dict[str, pd.DataFrame],
    ) -> pd.Series:
        """
        Calculate a scorecard score for each row in df.

        Requires WoE columns ('{variable}_woe') to be present.
        Returns a Series of integer scores, indexed as df.
        """
        if self.scorecard_table is None:
            self.build(model, bin_stats)

        # Build woe → points lookup per variable
        scores = pd.Series(
            float(self.A), index=df.index, name="score"
        )

        for variable in model.variables:
            woe_col = f"{variable}_woe"
            if woe_col not in df.columns:
                raise KeyError(
                    f"Column '{woe_col}' not found in dataframe. "
                    "Run BinningPipeline.transform() first."
                )

            var_table    = self.scorecard_table[
                self.scorecard_table["variable"] == variable
            ][["woe", "points"]].drop_duplicates("woe")

            woe_to_points = dict(zip(var_table["woe"], var_table["points"]))

            # Round WoE to 4dp to match scorecard table precision
            mapped_woe = df[woe_col].round(4)
            points     = mapped_woe.map(woe_to_points)

            if points.isna().any():
                unmapped = mapped_woe[points.isna()].dropna().unique().tolist()
                preview  = sorted(unmapped)[:10]
                raise ValueError(
                    f"Unable to map {int(points.isna().sum())} row(s) in "
                    f"'{woe_col}' to scorecard points. "
                    f"Unmapped rounded WoE values: {preview}. "
                    "Check for missing WoE values, unseen categories, or "
                    "scorecard table rounding collisions."
                )

            scores += points

        return scores.round().astype(int)

    # ------------------------------------------------------------------
    # Score → PD conversion
    # ------------------------------------------------------------------

    def score_to_pd(self, score: float) -> float:
        """Convert a scorecard score to an approximate PD."""
        log_odds = (self.A - score) / self.B
        return 1 / (1 + np.exp(-log_odds))

    def pd_to_score(self, pd_value: float) -> float:
        """Convert a PD to an approximate scorecard score."""
        log_odds = np.log(pd_value / (1 - pd_value))
        return self.A - self.B * log_odds

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> str:
        """Print-ready representation of the scorecard."""
        if self.scorecard_table is None:
            return "Scorecard not built yet. Call build() first."

        lines = [
            "=" * 72,
            f"SCORECARD",
            f"  PDO={self.pdo}  |  Base Score={self.base_score}  "
            f"|  Base Odds={self.base_odds}:1 (goods:bads)",
            f"  Scaling constants: A={self.A:.2f}, B={self.B:.2f}",
            "=" * 72,
        ]

        for variable in self.scorecard_table["variable"].unique():
            lines.append(f"\n  {variable.upper().replace('_', ' ')}")
            lines.append("  " + "-" * 55)
            var_rows = self.scorecard_table[
                self.scorecard_table["variable"] == variable
            ]
            for _, row in var_rows.iterrows():
                lines.append(
                    f"    {str(row['bin']):<28}  "
                    f"WoE: {row['woe']:>7.4f}  "
                    f"Bad Rate: {row['bad_rate']:>6.2%}  "
                    f"Points: {row['points']:>5}"
                )

        lines += [
            "",
            f"  {'OFFSET (A)':<38} Points: {self.A:>7.2f}",
            "=" * 72,
            "",
            "  Score → PD reference:",
            "  " + "-" * 40,
        ]
        for score in [500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700]:
            lines.append(
                f"    Score {score}  →  PD ≈ {self.score_to_pd(score):.2%}"
            )
        lines.append("=" * 72)

        return "\n".join(lines)
