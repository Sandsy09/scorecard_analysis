"""
modelling/logistic_model.py

Logistic regression fitted on WoE-transformed variables for scorecard
development, with built-in diagnostics covering:

    - Coefficient signs       (must all be positive post-WoE transform)
    - Coefficient proximity   (should be ~ 1.0 after WoE transform)
    - VIF                     (multicollinearity check)
    - Statistical significance (p-values)

Theory recap:
    After WoE transformation, the logistic regression equation is:

        log-odds = β0 + β1·WoE_income + β2·WoE_LTV + ...

    Because WoE is already in log-odds units, each βj should be ≈ 1.0.
    A coefficient >> 1.0 or << 1.0 suggests something is wrong:
        - Binning may need revision
        - Multicollinearity may be distorting coefficients

    Negative coefficient on a WoE variable is a red flag:
        - High WoE = high risk → should increase log-odds
        - Negative β means the model is predicting the opposite direction
        - Almost always caused by multicollinearity
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Optional


class ScorecardLogisticRegression:
    """
    Logistic regression for WoE-transformed scorecard variables.

    Uses statsmodels for full coefficient diagnostics (p-values,
    standard errors, confidence intervals) alongside VIF checks.

    After fitting, call predict_proba() to get PD estimates.
    """

    VIF_WARNING_THRESHOLD = 5.0   # flag for review
    VIF_ACTION_THRESHOLD  = 10.0  # strong action recommended
    COEF_DEVIATION_WARN   = 0.5   # flag if |β - 1.0| > this value

    def __init__(
        self,
        variables: List[str],
        target: str,
    ):
        """
        Parameters
        ----------
        variables : list of variable names (without '_woe' suffix)
        target    : binary target column name
        """
        self.variables   = variables
        self.target      = target
        self.woe_cols    = [f"{v}_woe" for v in variables]

        # Set after fitting
        self.sm_model    = None
        self.intercept:  Optional[float]      = None
        self.coefficients: Optional[pd.Series] = None
        self.diagnostics: Dict                = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ScorecardLogisticRegression":
        """
        Fit the model on a WoE-transformed dataframe.
        df must contain '{variable}_woe' columns and the target column.
        """
        missing_cols = [c for c in self.woe_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing WoE columns in dataframe: {missing_cols}. "
                "Ensure BinningPipeline.transform() has been called."
            )

        X = df[self.woe_cols].copy()
        y = df[self.target]

        # Fit via statsmodels to access full diagnostics
        X_const   = sm.add_constant(X)
        self.sm_model = sm.Logit(y, X_const).fit(disp=0)

        self.intercept    = float(self.sm_model.params["const"])
        self.coefficients = pd.Series(
            self.sm_model.params[self.woe_cols].values,
            index=self.variables,
        )

        self._run_diagnostics(X)
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _run_diagnostics(self, X: pd.DataFrame) -> None:
        # 1. Coefficient signs
        negative_coefs = self.coefficients[self.coefficients < 0]

        # 2. Coefficient deviation from 1.0
        deviations     = (self.coefficients - 1.0).abs()
        large_dev      = deviations[deviations > self.COEF_DEVIATION_WARN]

        # 3. VIF
        # variance_inflation_factor regresses each variable against all others.
        # With only one variable there are no others to regress against —
        # the function receives a zero-column matrix and raises a ValueError.
        # VIF is also meaningless with a single variable (multicollinearity
        # requires at least two), so we skip it and return a clear placeholder.
        if X.shape[1] < 2:
            vif_df = pd.DataFrame({
                "variable": X.columns,
                "vif":      [np.nan],
            })
        else:
            vif_df = pd.DataFrame({
                "variable": X.columns,
                "vif":      [
                    variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])
                ],
            })
        high_vif = vif_df[vif_df["vif"] > self.VIF_WARNING_THRESHOLD]

        # 4. P-values
        p_values        = self.sm_model.pvalues[self.woe_cols].copy()
        p_values.index  = self.variables
        insignificant   = p_values[p_values > 0.05]

        self.diagnostics = {
            "negative_coefficients": negative_coefs,
            "coefficient_deviations": deviations,
            "large_deviations":       large_dev,
            "vif":                    vif_df,
            "high_vif":               high_vif,
            "p_values":               p_values,
            "insignificant":          insignificant,
            "warnings":               self._build_warnings(
                negative_coefs, large_dev, high_vif, insignificant, vif_df
            ),
        }

    def _build_warnings(
        self,
        negative_coefs: pd.Series,
        large_dev: pd.Series,
        high_vif: pd.DataFrame,
        insignificant: pd.Series,
        vif_df: pd.DataFrame,
    ) -> List[str]:
        warnings = []

        if not negative_coefs.empty:
            warnings.append(
                f"⚠️  Negative coefficient(s) post-WoE: "
                f"{negative_coefs.index.tolist()}. "
                "Likely caused by multicollinearity — check VIF and "
                "consider removing one of the correlated pair."
            )

        if not large_dev.empty:
            warnings.append(
                f"⚠️  Coefficient(s) deviating significantly from 1.0: "
                f"{large_dev.index.tolist()}. "
                "Review binning for these variables."
            )

        if not high_vif.empty:
            action_vars = high_vif[
                high_vif["vif"] > self.VIF_ACTION_THRESHOLD
            ]["variable"].tolist()
            warnings.append(
                f"⚠️  High VIF detected: "
                f"{high_vif.set_index('variable')['vif'].round(2).to_dict()}. "
                + (
                    f"Strong action recommended for: {action_vars}."
                    if action_vars else ""
                )
            )
        elif high_vif.empty and vif_df["vif"].isna().all():
            warnings.append(
                "ℹ️  VIF not calculated — only one variable in model. "
                "Multicollinearity check is not applicable."
            )

        if not insignificant.empty:
            warnings.append(
                f"⚠️  Insignificant variable(s) (p > 0.05): "
                f"{insignificant.index.tolist()}. "
                "Consider removing unless strongly justified by business logic."
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
        X       = df[self.woe_cols].copy()
        X_const = sm.add_constant(X, has_constant="add")
        return self.sm_model.predict(X_const).values

    # ------------------------------------------------------------------
    # Log-odds (used by scorecard scaler)
    # ------------------------------------------------------------------

    def log_odds(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(df)
        return np.log(p / (1 - p))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def coefficient_table(self) -> pd.DataFrame:
        """Summary table of coefficients, p-values, and VIF."""
        vif_series = (
            self.diagnostics["vif"]
            .set_index("variable")["vif"]
            .rename(index=lambda x: x.replace("_woe", ""))
        )
        return pd.DataFrame({
            "coefficient": self.coefficients.round(4),
            "p_value":     self.diagnostics["p_values"].round(4),
            "vif":         vif_series.round(2),
            "deviation_from_1": self.diagnostics["coefficient_deviations"].round(4),
        })

    def diagnostic_report(self) -> str:
        lines = [
            "=" * 65,
            "LOGISTIC REGRESSION DIAGNOSTICS",
            "=" * 65,
            f"  Intercept (β0): {self.intercept:.4f}",
            "",
            "  Coefficients:",
            self.coefficient_table().to_string(index=True),
            "",
            "  Warnings:",
        ]
        for w in self.diagnostics["warnings"]:
            lines.append(f"    {w}")
        lines.append("=" * 65)
        return "\n".join(lines)