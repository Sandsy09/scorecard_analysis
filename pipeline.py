"""
pipeline.py

End-to-end scorecard development pipeline for the PD_cust × f(deal) model.
Orchestrates all components in the correct order with clear logging.

Split strategy
--------------
Data splitting is controlled by a DataSplitConfig, which supports two modes:

    Date mode (recommended for production):
        Explicit start/end dates for each sample. Gives full control over
        which vintages are in each set and ensures OOT loans are sufficiently
        matured to have observable outcomes.

        config = DataSplitConfig.from_dates(
            dev_start  = "2020-01-01", dev_end  = "2021-12-31",
            oot_start  = "2022-01-01", oot_end  = "2022-12-31",
            # optionally:
            val_start  = "2021-07-01", val_end  = "2021-12-31",
        )

    Percentage mode (useful for early exploration):
        Pulls all data in a single query and splits chronologically by a
        date column. The last oot_pct of records (sorted ascending by date)
        become OOT; the preceding val_pct (if given) become validation.

        ⚠️  Warning: if your data runs to a recent date, the OOT tail may
        contain immature loans with artificially low bad rates. Always check
        the OOT bad rate and vintage distribution before trusting OOT metrics.

        config = DataSplitConfig.from_percentage(
            date_column = "application_date",
            data_start  = "2019-01-01",
            data_end    = "2022-12-31",
            oot_pct     = 0.20,
            val_pct     = 0.10,   # optional
        )

Validation set
--------------
An optional validation set sits chronologically between development and OOT:

    [────── Development ──────][── Validation ──][─── OOT ───]
       Fit model & binning        Iterate safely    Final holdout

Use the validation set to iterate on binning decisions and review diagnostics
without contaminating OOT. OOT should remain untouched until the model is
finalised. When a validation set is present, validate() will report all three
samples side-by-side, with a Gini stability check across the chain.

Usage
-----
    config = DataSplitConfig.from_dates(
        dev_start = "2020-01-01", dev_end = "2021-06-30",
        val_start = "2021-07-01", val_end = "2021-12-31",
        oot_start = "2022-01-01", oot_end = "2022-12-31",
    )

    pipeline = ScorecardPipeline(
        connection_string  = "mssql+pyodbc://server/database?...",
        target             = "default_flag",
        customer_variables = ["annual_income", "credit_bureau_score",
                              "employment_status", "months_at_address"],
        deal_variables     = ["ltv_ratio", "loan_term_months",
                              "deposit_pct", "vehicle_age_years"],
    )

    pipeline.run_full_pipeline(
        split_config        = config,
        strata_variable     = "credit_bureau_score",
        customer_cut_points = {"annual_income": [15000, 25000, 40000]},
        deal_cut_points     = {"ltv_ratio":     [60, 80, 100]},
    )
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from data.extractor import DataExtractor
from modelling.logistic_model import ScorecardLogisticRegression
from modelling.scorecard_scaler import ScorecardScaler
from preprocessing.binning import BinningPipeline
from testing.statistical_tests import (
    ContingencyTable,
    DealVariableDiagnostics,
    InteractionTestingPipeline,
    PairwiseBreslowDay,
    TestDecision,
)
from validation.metrics import ValidationReport


# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------

@dataclass
class DataSplitConfig:
    """
    Encapsulates the data splitting strategy for the pipeline.

    Do not instantiate directly — use the factory classmethods:
        DataSplitConfig.from_dates(...)
        DataSplitConfig.from_percentage(...)

    Attributes
    ----------
    mode        : 'date' or 'percentage'
    dev_start   : development period start (date mode)
    dev_end     : development period end (date mode)
    val_start   : validation period start (date mode, optional)
    val_end     : validation period end (date mode, optional)
    oot_start   : OOT period start (date mode)
    oot_end     : OOT period end (date mode)
    date_column : column to sort by for chronological split (percentage mode)
    data_start  : earliest application date to pull (percentage mode)
    data_end    : latest application date to pull (percentage mode)
    oot_pct     : proportion of data reserved for OOT (percentage mode)
    val_pct     : proportion of data reserved for validation (percentage mode,
                  optional). Taken from the period immediately before OOT.
    """

    mode: Literal["date", "percentage"]

    # Date mode fields
    dev_start: Optional[str] = None
    dev_end:   Optional[str] = None
    val_start: Optional[str] = None
    val_end:   Optional[str] = None
    oot_start: Optional[str] = None
    oot_end:   Optional[str] = None

    # Percentage mode fields
    date_column: Optional[str] = None
    data_start:  Optional[str] = None
    data_end:    Optional[str] = None
    oot_pct:     float         = 0.20
    val_pct:     Optional[float] = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dates(
        cls,
        dev_start: str,
        dev_end:   str,
        oot_start: str,
        oot_end:   str,
        val_start: Optional[str] = None,
        val_end:   Optional[str] = None,
    ) -> "DataSplitConfig":
        """
        Date-based split. Recommended for production use where you have
        explicit, governance-documented period boundaries.

        Parameters
        ----------
        dev_start / dev_end : development (training) period
        oot_start / oot_end : out-of-time holdout period
        val_start / val_end : optional validation period — must sit
                              chronologically between dev_end and oot_start
        """
        if val_start is not None and val_end is None:
            raise ValueError("val_end must be provided when val_start is set.")
        if val_end is not None and val_start is None:
            raise ValueError("val_start must be provided when val_end is set.")

        return cls(
            mode      = "date",
            dev_start = dev_start,
            dev_end   = dev_end,
            val_start = val_start,
            val_end   = val_end,
            oot_start = oot_start,
            oot_end   = oot_end,
        )

    @classmethod
    def from_percentage(
        cls,
        date_column: str,
        data_start:  str,
        data_end:    str,
        oot_pct:     float          = 0.20,
        val_pct:     Optional[float] = None,
    ) -> "DataSplitConfig":
        """
        Percentage-based chronological split. Pulls all data in one query
        then divides by record count after sorting ascending by date_column.

        Parameters
        ----------
        date_column : column used to sort records chronologically
        data_start  : start of the full data pull
        data_end    : end of the full data pull
        oot_pct     : proportion assigned to OOT (taken from the end)
        val_pct     : proportion assigned to validation (taken just before OOT)

        ⚠️  If data_end is recent, OOT loans may not be sufficiently matured.
            Check the OOT bad rate and vintage spread before relying on OOT metrics.
        """
        if not (0 < oot_pct < 1):
            raise ValueError("oot_pct must be between 0 and 1 (exclusive).")
        if val_pct is not None:
            if not (0 < val_pct < 1):
                raise ValueError("val_pct must be between 0 and 1 (exclusive).")
            if oot_pct + val_pct >= 1:
                raise ValueError(
                    f"oot_pct ({oot_pct}) + val_pct ({val_pct}) must be < 1.0 "
                    "to leave records for development."
                )

        return cls(
            mode        = "percentage",
            date_column = date_column,
            data_start  = data_start,
            data_end    = data_end,
            oot_pct     = oot_pct,
            val_pct     = val_pct,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_validation(self) -> bool:
        """True when a validation set has been configured."""
        if self.mode == "date":
            return self.val_start is not None
        return self.val_pct is not None

    def summary(self) -> str:
        """Human-readable description of the configured split."""
        lines = [f"Split mode: {self.mode.upper()}"]
        if self.mode == "date":
            lines.append(f"  Development : {self.dev_start} → {self.dev_end}")
            if self.has_validation:
                lines.append(f"  Validation  : {self.val_start} → {self.val_end}")
            lines.append(f"  OOT         : {self.oot_start} → {self.oot_end}")
        else:
            lines.append(f"  Data range  : {self.data_start} → {self.data_end}")
            lines.append(f"  Date column : {self.date_column}")
            dev_pct = 1.0 - self.oot_pct - (self.val_pct or 0.0)
            lines.append(f"  Development : {dev_pct:.0%} (earliest records)")
            if self.has_validation:
                lines.append(f"  Validation  : {self.val_pct:.0%}")
            lines.append(f"  OOT         : {self.oot_pct:.0%} (latest records)")
            lines.append(
                "  ⚠️  Check OOT bad rate — recent records may be immature."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ScorecardPipeline:
    """
    Full end-to-end pipeline for the PD_cust × f(deal) scorecard.

    Steps
    -----
    1. extract_data()            — split data according to DataSplitConfig
    2. run_binning()             — fit WoE bins for customer and deal variables
    3. run_interaction_testing() — Breslow-Day → CMH flow for deal variables
    4. fit_models()              — logistic regression with diagnostics
    5. build_scorecard()         — scale coefficients to points
    6. validate()                — Gini, KS, HL, PSI, CSI across all samples

    Each step returns self so they can be chained or called individually.
    """

    def __init__(
        self,
        connection_string:  str,
        target:             str            = "default_flag",
        customer_variables: Optional[List[str]] = None,
        deal_variables:     Optional[List[str]] = None,
        pdo:        float = 20,
        base_score: float = 600,
        base_odds:  float = 50,
    ):
        self.connection_string  = connection_string
        self.target             = target
        self.customer_variables = customer_variables or []
        self.deal_variables     = deal_variables     or []

        # Components — each is a reusable class
        self.extractor          = DataExtractor(connection_string)
        self.customer_binner    = BinningPipeline(target)
        self.deal_binner        = BinningPipeline(target)
        self.interaction_tester = InteractionTestingPipeline()
        self.deal_diagnostics   = DealVariableDiagnostics()
        self.scaler             = ScorecardScaler(pdo, base_score, base_odds)
        self.validator          = ValidationReport("PD_cust × f(deal) Scorecard")

        self.customer_model: Optional[ScorecardLogisticRegression] = None
        self.deal_model:     Optional[ScorecardLogisticRegression] = None

        # Data stores — val_* are None when no validation set is configured
        self.dev_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.oot_data: Optional[pd.DataFrame] = None
        self.dev_woe:  Optional[pd.DataFrame] = None
        self.val_woe:  Optional[pd.DataFrame] = None
        self.oot_woe:  Optional[pd.DataFrame] = None

        # Stores ContingencyTables built during interaction testing
        # so PairwiseBreslowDay can reuse them in run_diagnostics()
        self._interaction_tables: Dict[str, List[ContingencyTable]] = {}

        # Split config stored for reference / reporting
        self._split_config: Optional[DataSplitConfig] = None

    # ------------------------------------------------------------------
    # Step 1: Data extraction and splitting
    # ------------------------------------------------------------------

    def extract_data(self, split_config: DataSplitConfig) -> "ScorecardPipeline":
        """
        Extract and split data according to a DataSplitConfig.

        Date mode   : runs one query per period (dev, val, OOT).
        Percentage  : runs a single query for the full range, then
                      splits chronologically in Python.

        Parameters
        ----------
        split_config : DataSplitConfig instance (use factory classmethods)
        """
        self._log("STEP 1: Extracting and splitting data")
        self._log(f"\n{split_config.summary()}\n")
        self._split_config = split_config

        if split_config.mode == "date":
            self._extract_by_dates(split_config)
        else:
            self._extract_by_percentage(split_config)

        self._log_sample_summary()
        return self

    def _extract_by_dates(self, cfg: DataSplitConfig) -> None:
        """Pull each period separately using its explicit date range."""
        with self.extractor as db:
            self.dev_data = db.get_combined_data(cfg.dev_start, cfg.dev_end)
            self.oot_data = db.get_combined_data(cfg.oot_start, cfg.oot_end)
            if cfg.has_validation:
                self.val_data = db.get_combined_data(cfg.val_start, cfg.val_end)

    def _extract_by_percentage(self, cfg: DataSplitConfig) -> None:
        """
        Pull all data in one query, sort chronologically by date_column,
        then divide into dev / val / OOT by record count.

        Chronological sort means the oldest records train the model and
        the most recent records form the OOT holdout — matching how the
        model will be used in deployment.
        """
        if cfg.date_column is None:
            raise ValueError(
                "date_column must be set for percentage-mode splits. "
                "Use DataSplitConfig.from_percentage(date_column=...)."
            )

        with self.extractor as db:
            all_data = db.get_combined_data(cfg.data_start, cfg.data_end)

        if cfg.date_column not in all_data.columns:
            raise ValueError(
                f"date_column '{cfg.date_column}' not found in extracted data. "
                f"Available columns: {all_data.columns.tolist()}"
            )

        # Sort ascending so oldest → newest
        all_data = all_data.sort_values(cfg.date_column).reset_index(drop=True)
        n        = len(all_data)

        n_oot = int(np.floor(n * cfg.oot_pct))
        n_val = int(np.floor(n * cfg.val_pct)) if cfg.val_pct is not None else 0
        n_dev = n - n_oot - n_val

        if n_dev <= 0:
            raise ValueError(
                f"Percentage split leaves no development records. "
                f"n={n}, oot_pct={cfg.oot_pct}, val_pct={cfg.val_pct}. "
                "Reduce oot_pct or val_pct."
            )

        self.dev_data = all_data.iloc[:n_dev].copy()
        if n_val > 0:
            self.val_data = all_data.iloc[n_dev : n_dev + n_val].copy()
        self.oot_data = all_data.iloc[n_dev + n_val :].copy()

        # Warn if OOT bad rate looks suspiciously low (potential immaturity)
        oot_bad_rate = self.oot_data[self.target].mean()
        dev_bad_rate = self.dev_data[self.target].mean()
        if oot_bad_rate < dev_bad_rate * 0.5:
            warnings.warn(
                f"OOT bad rate ({oot_bad_rate:.2%}) is less than half the "
                f"development bad rate ({dev_bad_rate:.2%}). "
                "OOT records may not be sufficiently matured — "
                "OOT performance metrics should be interpreted with caution.",
                UserWarning,
                stacklevel=2,
            )

    def _log_sample_summary(self) -> None:
        """Print a summary table of each sample after splitting."""
        samples = [("Development", self.dev_data)]
        if self.val_data is not None:
            samples.append(("Validation", self.val_data))
        samples.append(("OOT", self.oot_data))

        total = sum(len(s) for _, s in samples)
        self._log(f"\n  {'Sample':<15} {'Records':>10}  {'% Total':>8}  {'Bad Rate':>9}")
        self._log(f"  {'-'*15} {'-'*10}  {'-'*8}  {'-'*9}")
        for label, df in samples:
            self._log(
                f"  {label:<15} {len(df):>10,}  "
                f"{len(df)/total:>8.1%}  "
                f"{df[self.target].mean():>9.2%}"
            )
        self._log(f"  {'TOTAL':<15} {total:>10,}")

    # ------------------------------------------------------------------
    # Step 2: Binning
    # ------------------------------------------------------------------

    def run_binning(
        self,
        customer_cut_points: Optional[Dict[str, List[float]]] = None,
        deal_cut_points:     Optional[Dict[str, List[float]]] = None,
        customer_var_types:  Optional[Dict[str, str]] = None,
        deal_var_types:      Optional[Dict[str, str]] = None,
    ) -> "ScorecardPipeline":
        """
        Fit WoE bins on the development sample and apply to all samples.

        Bins are always fitted on development data only. Validation and OOT
        samples receive the same bin mappings via transform() — never refit
        on those samples, as this would constitute data leakage.

        customer_cut_points : dict of {variable: [cut1, cut2, ...]}
                              for manual binning. Variables not listed
                              use equal-frequency auto-binning.
        customer_var_types  : dict of {variable: 'categorical'} for
                              categorical predictors. Default is continuous.
        """
        self._log("STEP 2: Binning variables")

        ccp = customer_cut_points or {}
        dcp = deal_cut_points     or {}
        cvt = customer_var_types  or {}
        dvt = deal_var_types      or {}

        for var in self.customer_variables:
            self.customer_binner.add_variable(
                var,
                variable_type = cvt.get(var, "continuous"),
                cut_points    = ccp.get(var),
            )

        for var in self.deal_variables:
            self.deal_binner.add_variable(
                var,
                variable_type = dvt.get(var, "continuous"),
                cut_points    = dcp.get(var),
            )

        # Fit on development only
        self.customer_binner.fit(self.dev_data)
        self.deal_binner.fit(self.dev_data)

        self._log("\n  Customer Variables:")
        self.customer_binner.print_iv_summary()
        self._log("\n  Deal Variables:")
        self.deal_binner.print_iv_summary()

        # Apply WoE transform to all available samples
        self.dev_woe = self._apply_woe(self.dev_data)
        self.oot_woe = self._apply_woe(self.oot_data)
        if self.val_data is not None:
            self.val_woe = self._apply_woe(self.val_data)

        return self

    def _apply_woe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply both customer and deal WoE transforms to a dataframe."""
        out = self.customer_binner.transform(df)
        out = self.deal_binner.transform(out)
        return out

    # ------------------------------------------------------------------
    # Step 3: Interaction testing (Breslow-Day → CMH)
    # ------------------------------------------------------------------

    def run_interaction_testing(
        self,
        strata_variable: str,
        n_strata: int = 3,
    ) -> "ScorecardPipeline":
        """
        Test whether deal variables have a consistent effect on default
        across customer risk strata (validates the multiplicative structure).

        Always runs on the development sample only. Interaction testing on
        validation or OOT data is not appropriate — those samples are for
        evaluating a fixed model, not for structural assumption testing.

        strata_variable : variable used to define customer risk strata
        n_strata        : number of strata (default 3: Low / Medium / High)
        """
        self._log("STEP 3: Interaction testing (Breslow-Day → CMH)")

        df            = self.dev_data.copy()
        strata_labels = ["Low", "Medium", "High"][:n_strata]

        df["_stratum"] = pd.qcut(
            df[strata_variable], q=n_strata, labels=strata_labels
        )

        for deal_var in self.deal_variables:
            median            = df[deal_var].median()
            df["_deal_high"]  = (df[deal_var] > median).astype(int)

            tables = []
            for stratum in strata_labels:
                s = df[df["_stratum"] == stratum]
                tables.append(ContingencyTable(
                    a            = int(((s["_deal_high"] == 1) & (s[self.target] == 1)).sum()),
                    b            = int(((s["_deal_high"] == 1) & (s[self.target] == 0)).sum()),
                    c            = int(((s["_deal_high"] == 0) & (s[self.target] == 1)).sum()),
                    d            = int(((s["_deal_high"] == 0) & (s[self.target] == 0)).sum()),
                    stratum_name = stratum,
                ))

            self._interaction_tables[deal_var] = tables
            self.interaction_tester.run_variable(deal_var, tables)

        self.interaction_tester.print_results()
        return self

    # ------------------------------------------------------------------
    # Step 3b: Stratum diagnostics
    # ------------------------------------------------------------------

    def run_diagnostics(
        self,
        strata_variable:   str,
        n_strata:          int  = 3,
        min_cell_threshold: int = 5,
        verbose:           bool = True,
    ) -> "ScorecardPipeline":
        """
        Granular stratum-level diagnostics for deal variables after a
        Breslow-Day failure. See original pipeline docstring for full detail.
        """
        self._log("STEP 3b: Running stratum diagnostics on deal variables")

        if self.dev_data is None:
            raise RuntimeError(
                "No data available. Run extract_data() and run_binning() first."
            )
        if not self._interaction_tables:
            raise RuntimeError(
                "No interaction tables found. "
                "Run run_interaction_testing() before run_diagnostics()."
            )

        failed_vars = [
            var for var, res in self.interaction_tester._results.items()
            if res.breslow_day.decision == TestDecision.FAIL
        ]

        if failed_vars:
            self._log(
                f"\n  Variables with BD FAIL: {failed_vars}"
                "\n  Running pairwise Breslow-Day..."
            )
            for var in failed_vars:
                tables = self._interaction_tables.get(var)
                if not tables:
                    continue

                print(f"\n{'=' * 65}")
                print(f"PAIRWISE BRESLOW-DAY: {var}")
                print(f"{'=' * 65}")

                pairwise_result = PairwiseBreslowDay(tables).run()
                print(pairwise_result.to_string(index=False))

                significant = pairwise_result[
                    pairwise_result["bd_p_value"] < self.interaction_tester.alpha
                ]
                if significant.empty:
                    print(
                        "\n  No individual pairs significant — global BD failure "
                        "may be cumulative or driven by sparse cells."
                    )
                else:
                    pairs = list(zip(significant["stratum_a"], significant["stratum_b"]))
                    print(f"\n  ⚠️  Significant pairs: {pairs}")
        else:
            self._log("  No BD failures — pairwise testing skipped.")

        df            = self.dev_data.copy()
        strata_labels = ["Low", "Medium", "High"][:n_strata]
        df["_stratum"] = pd.qcut(
            df[strata_variable], q=n_strata, labels=strata_labels
        )

        df_with_woe   = self.deal_binner.transform(df)
        woe_deal_cols = [f"{v}_woe" for v in self.deal_variables]

        self.deal_diagnostics = DealVariableDiagnostics(
            min_cell_threshold=min_cell_threshold
        )
        self.deal_diagnostics.run(
            df           = df_with_woe,
            deal_vars    = woe_deal_cols,
            customer_var = "_stratum",
            outcome_var  = self.target,
            verbose      = verbose,
        )
        return self

    # ------------------------------------------------------------------
    # Step 4: Fit models
    # ------------------------------------------------------------------

    def fit_models(
        self,
        min_iv: float = 0.10,
        max_iv: float = 0.50,
    ) -> "ScorecardPipeline":
        """
        Fit logistic regression on WoE-transformed development data.
        Variables are selected based on IV range.
        """
        self._log("STEP 4: Fitting logistic regression models")

        selected_customer = self.customer_binner.get_selected_variables(min_iv, max_iv)
        selected_deal     = self.deal_binner.get_selected_variables(min_iv, max_iv)

        self._log(f"  Selected customer variables: {selected_customer}")
        self._log(f"  Selected deal variables:     {selected_deal}")

        if not selected_customer:
            raise RuntimeError(
                "No customer variables passed the IV filter. "
                "Review binning or adjust min_iv / max_iv thresholds."
            )

        self._log("\n  Fitting PD_cust model...")
        self.customer_model = ScorecardLogisticRegression(
            selected_customer, self.target
        ).fit(self.dev_woe)
        print(self.customer_model.diagnostic_report())

        if selected_deal:
            self._log("\n  Fitting f(deal) model...")
            self.deal_model = ScorecardLogisticRegression(
                selected_deal, self.target
            ).fit(self.dev_woe)
            print(self.deal_model.diagnostic_report())
        else:
            self._log("  ⚠️  No deal variables passed IV filter — "
                      "f(deal) component not fitted.")

        return self

    # ------------------------------------------------------------------
    # Step 5: Build scorecard
    # ------------------------------------------------------------------

    def build_scorecard(self) -> "ScorecardPipeline":
        self._log("STEP 5: Building scorecard")

        if self.customer_model is None:
            raise RuntimeError("Fit models before building scorecard.")

        bin_stats = self.customer_binner.get_all_bin_stats()
        self.scaler.build(self.customer_model, bin_stats)
        print(self.scaler.display())
        return self

    # ------------------------------------------------------------------
    # Step 6: Validation
    # ------------------------------------------------------------------

    def validate(self) -> "ScorecardPipeline":
        """
        Run the full validation suite across all available samples.

        When a validation set is present, reports dev / val / OOT side-by-side
        and flags any Gini drop exceeding 5 points across the chain:
            dev → val  (should be small — same population, held-out slice)
            val → OOT  (the key temporal stability check)
            dev → OOT  (overall drop reported for governance)
        """
        self._log("STEP 6: Model validation")

        if self.customer_model is None:
            raise RuntimeError("Fit models before running validation.")

        bin_stats      = self.customer_binner.get_all_bin_stats()
        y_pred_dev     = self.customer_model.predict_proba(self.dev_woe)
        y_pred_oot     = self.customer_model.predict_proba(self.oot_woe)
        scores_dev     = self.scaler.score(self.dev_woe, self.customer_model, bin_stats)
        scores_oot     = self.scaler.score(self.oot_woe, self.customer_model, bin_stats)

        y_pred_val: Optional[np.ndarray] = None
        scores_val: Optional[pd.Series]  = None
        if self.val_woe is not None:
            y_pred_val = self.customer_model.predict_proba(self.val_woe)
            scores_val = self.scaler.score(self.val_woe, self.customer_model, bin_stats)

        self.validator.run(
            y_true_dev = self.dev_data[self.target],
            y_pred_dev = pd.Series(y_pred_dev),
            scores_dev = scores_dev,
            y_true_oot = self.oot_data[self.target],
            y_pred_oot = pd.Series(y_pred_oot),
            scores_oot = scores_oot,
            vars_dev   = self.dev_woe[self.customer_model.woe_cols],
            vars_oot   = self.oot_woe[self.customer_model.woe_cols],
            variables  = self.customer_model.variables,
            # Validation set — passed as None when not configured
            y_true_val = self.val_data[self.target] if self.val_data is not None else None,
            y_pred_val = pd.Series(y_pred_val) if y_pred_val is not None else None,
            scores_val = scores_val,
            vars_val   = self.val_woe[self.customer_model.woe_cols] if self.val_woe is not None else None,
        )
        self.validator.print_report()
        return self

    # ------------------------------------------------------------------
    # Convenience: run entire pipeline in one call
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        split_config:        DataSplitConfig,
        strata_variable:     str  = "credit_bureau_score",
        customer_cut_points: Optional[Dict] = None,
        deal_cut_points:     Optional[Dict] = None,
        customer_var_types:  Optional[Dict] = None,
        deal_var_types:      Optional[Dict] = None,
        min_iv: float = 0.10,
        max_iv: float = 0.50,
    ) -> "ScorecardPipeline":
        """
        Run all pipeline steps end-to-end.

        Examples
        --------
        # Date-based split with validation set
        config = DataSplitConfig.from_dates(
            dev_start = "2020-01-01", dev_end = "2021-06-30",
            val_start = "2021-07-01", val_end = "2021-12-31",
            oot_start = "2022-01-01", oot_end = "2022-12-31",
        )
        pipeline.run_full_pipeline(config, strata_variable="credit_bureau_score")

        # Percentage-based split (exploration / quick iteration)
        config = DataSplitConfig.from_percentage(
            date_column = "application_date",
            data_start  = "2019-01-01",
            data_end    = "2022-12-31",
            oot_pct     = 0.20,
            val_pct     = 0.10,
        )
        pipeline.run_full_pipeline(config, strata_variable="credit_bureau_score")
        """
        return (
            self
            .extract_data(split_config)
            .run_binning(
                customer_cut_points, deal_cut_points,
                customer_var_types, deal_var_types,
            )
            .run_interaction_testing(strata_variable)
            .fit_models(min_iv, max_iv)
            .build_scorecard()
            .validate()
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _log(message: str) -> None:
        print(message)