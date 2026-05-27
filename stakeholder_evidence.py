"""
stakeholder_evidence.py

Produces four pieces of stakeholder evidence to explain why
interaction term point contributions can appear counterintuitive
when the Breslow-Day test has confirmed a genuine interaction effect.

The four evidence pieces address a specific and common pushback:
    "The BD test showed high deposit reduces default even for low-Equifax
     customers — so why does the model score them lower for high deposit?"

Evidence produced:
    1. bad_rate_grid()         — raw bad rate and bads by Equifax × deal band.
                                 Validates the BD finding from first principles.

    2. interaction_pattern()   — bad rate by deal bin, separately per Equifax
                                 stratum. Shows that the protective effect of
                                 deposit exists in all strata but differs in
                                 magnitude — the core interaction finding.

    3. total_score_grid()      — mean total model score by Equifax × deal band.
                                 Shows that the overall score still rank-orders
                                 customers sensibly, even when individual
                                 interaction term contributions look odd in
                                 isolation.

    4. bd_reconciliation()     — printed text reconciling the BD finding with
                                 the interaction model behaviour. The key
                                 message: BD found the *direction* is consistent
                                 but the *magnitude* differs — the model captures
                                 that magnitude difference.

Usage
-----
    from stakeholder_evidence import StakeholderEvidenceReport

    # Minimum — raw data evidence only (pieces 1 and 2)
    report = StakeholderEvidenceReport(
        df          = pipeline.dev_data,
        equifax_var = "equifax_score",
        deal_var    = "deposit_pct",
        target      = "default_flag",
    )
    report.print_report()

    # Full — include total score grid (piece 3) via fitted interaction model
    report = StakeholderEvidenceReport(
        df          = pipeline.dev_data,
        equifax_var = "equifax_score",
        deal_var    = "deposit_pct",
        target      = "default_flag",
        model       = interaction_pipeline.best_model,
    )
    report.print_report()
    fig = report.plot_interaction_pattern()
    fig.savefig("interaction_pattern.png", dpi=150, bbox_inches="tight")
"""

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


class StakeholderEvidenceReport:
    """
    Produces four pieces of stakeholder evidence for an interaction model.

    Parameters
    ----------
    df : pd.DataFrame
        Development dataset containing raw (pre-WoE) variables and the
        binary target. Typically pipeline.dev_data.

    equifax_var : str
        Column name for the raw Equifax (or equivalent credit bureau) score.

    deal_var : str
        Column name for the deal variable under investigation (e.g.
        'deposit_pct'). Raw values, not WoE-transformed.

    target : str
        Binary target column (1 = bad, 0 = good).

    equifax_n_bands : int
        Number of equal-frequency Equifax bands. Default 3 (Low / Mid / High).

    deal_n_bands : int
        Number of equal-frequency deal variable bands. Default 4 produces
        quartile-style Low / Mid-Low / Mid-High / High labels.

    equifax_labels : list of str, optional
        Custom labels for Equifax bands (length must match equifax_n_bands).
        Default: ["Low", "Mid", "High"] for n=3.

    deal_labels : list of str, optional
        Custom labels for deal variable bands (length must match deal_n_bands).
        Default: ["Low", "Mid-Low", "Mid-High", "High"] for n=4.

    model : fitted InteractionLogisticRegression, optional
        If provided, enables evidence piece 3 (total score grid) and the
        predicted vs actuals chart. Must expose a predict_proba(df) method.

    df_model : pd.DataFrame, optional
        The pre-transformed DataFrame the model can score directly —
        i.e. the data that has already been through WoE transformation and
        Equifax standardisation, matching the columns the model was trained on.
        Required when model is provided. Typically the dev_woe DataFrame from
        interaction_pipeline with equifax_std already added.
        Must be index-aligned with df so probabilities map back correctly.

    pdo : float
        Points to Double the Odds — used to convert log-odds to score for
        the total score grid. Default 20.

    base_score : float
        Base score parameter. Default 600.

    base_odds : float
        Goods-to-bads ratio at base_score. Default 50.
    """

    _DEFAULT_EQUIFAX_LABELS: Dict[int, List[str]] = {
        2: ["Low", "High"],
        3: ["Low", "Mid", "High"],
        4: ["Low", "Mid-Low", "Mid-High", "High"],
        5: ["Very Low", "Low", "Mid", "High", "Very High"],
    }

    _DEFAULT_DEAL_LABELS: Dict[int, List[str]] = {
        2: ["Low", "High"],
        3: ["Low", "Mid", "High"],
        4: ["Low", "Mid-Low", "Mid-High", "High"],
        5: ["Very Low", "Low", "Mid", "High", "Very High"],
    }

    def __init__(
        self,
        df:               pd.DataFrame,
        equifax_var:      str,
        deal_var:         str,
        target:           str,
        equifax_n_bands:  int                 = 3,
        deal_n_bands:     int                 = 4,
        equifax_labels:   Optional[List[str]] = None,
        deal_labels:      Optional[List[str]] = None,
        model                                 = None,
        df_model:         Optional[pd.DataFrame] = None,
        pdo:              float               = 20,
        base_score:       float               = 600,
        base_odds:        float               = 50,
    ) -> None:
        self.df          = df.copy()
        self.equifax_var = equifax_var
        self.deal_var    = deal_var
        self.target      = target
        self.model       = model
        self.df_model    = df_model  # pre-transformed; used for model scoring

        # Scorecard scaling constants — base_odds is goods:bads while
        # model log-odds are bad:good, so A uses a minus sign.
        self.B = pdo / np.log(2)
        self.A = base_score - self.B * np.log(base_odds)

        # Validate inputs
        for col in [equifax_var, deal_var, target]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in dataframe. "
                    f"Available: {df.columns.tolist()}"
                )

        if model is not None and df_model is None:
            warnings.warn(
                "model was provided but df_model was not. "
                "The model requires pre-transformed inputs (WoE columns + equifax_std). "
                "Pass df_model=<your WoE-transformed dev DataFrame> to enable "
                "Evidence 3 (total score grid) and the predicted vs actuals chart. "
                "These outputs will be skipped until df_model is supplied."
            )
            self.model = None  # disable scoring to avoid silent failures

        # Build Equifax bands
        self._equifax_labels = (
            equifax_labels
            or self._DEFAULT_EQUIFAX_LABELS.get(
                equifax_n_bands,
                [f"Band {i+1}" for i in range(equifax_n_bands)],
            )
        )
        if len(self._equifax_labels) != equifax_n_bands:
            raise ValueError(
                f"equifax_labels length ({len(self._equifax_labels)}) "
                f"must match equifax_n_bands ({equifax_n_bands})."
            )

        # Build deal variable bands
        self._deal_labels = (
            deal_labels
            or self._DEFAULT_DEAL_LABELS.get(
                deal_n_bands,
                [f"Band {i+1}" for i in range(deal_n_bands)],
            )
        )
        if len(self._deal_labels) != deal_n_bands:
            raise ValueError(
                f"deal_labels length ({len(self._deal_labels)}) "
                f"must match deal_n_bands ({deal_n_bands})."
            )

        self.df["_equifax_band"] = pd.qcut(
            self.df[equifax_var],
            q=equifax_n_bands,
            labels=self._equifax_labels,
            duplicates="drop",
        )

        self.df["_deal_band"] = pd.qcut(
            self.df[deal_var],
            q=deal_n_bands,
            labels=self._deal_labels,
            duplicates="drop",
        )

        # Score if model is available
        if self.model is not None:
            self._attach_scores()

    # ------------------------------------------------------------------
    # Internal: score the dataframe using the interaction model
    # ------------------------------------------------------------------

    def _attach_scores(self) -> None:
        """
        Score using df_model (pre-transformed) and attach results to self.df.

        Probabilities and scores are computed on df_model then joined back to
        self.df on index, keeping the raw variable columns intact for banding.
        Stores:
            self.df['_score']  — scorecard scale score (A - B × log-odds)
            self.df['_proba']  — predicted probability of default
        """
        try:
            proba    = self.model.predict_proba(self.df_model)
            log_odds = np.log(np.clip(proba, 1e-9, 1 - 1e-9) / (1 - np.clip(proba, 1e-9, 1 - 1e-9)))

            # Align back to self.df via index — both must share the same index
            score_series = pd.Series(
                self.A - self.B * log_odds,
                index=self.df_model.index,
            )
            proba_series = pd.Series(proba, index=self.df_model.index)

            self.df["_score"] = self.df.index.map(score_series)
            self.df["_proba"] = self.df.index.map(proba_series)

        except Exception as exc:
            warnings.warn(
                f"Could not score dataframe using provided model: {exc}. "
                "Evidence 3 (total score grid) and the predicted vs actuals "
                "chart will be skipped. Check that df_model contains the "
                "columns the model expects (equifax_std, WoE deal columns, etc.)."
            )
            self.model = None

    # ------------------------------------------------------------------
    # Evidence piece 1: Bad rate grid
    # ------------------------------------------------------------------

    def bad_rate_grid(self) -> pd.DataFrame:
        """
        Cross-tabulation of Equifax band × deal variable band showing:
            N           — number of customers
            bads        — number of defaults
            bad_rate    — observed default rate
            pct_of_pop  — % of total population in this cell

        This is the raw data starting point. It validates the Breslow-Day
        finding without any model involvement — stakeholders can verify
        the pattern directly from the numbers.

        Returns
        -------
        pd.DataFrame with one row per Equifax band × deal band combination,
        sorted by Equifax band then deal band.
        """
        grp = (
            self.df
            .groupby(["_equifax_band", "_deal_band"], observed=True)
            .agg(
                N       =(self.target, "count"),
                bads    =(self.target, "sum"),
            )
            .reset_index()
        )
        grp["bad_rate"]   = (grp["bads"]  / grp["N"]).round(4)
        grp["pct_of_pop"] = (grp["N"] / len(self.df) * 100).round(1)

        grp = grp.rename(columns={
            "_equifax_band": "equifax_band",
            "_deal_band":    "deal_band",
        })

        # Preserve label ordering (low → high) via Categorical
        grp["equifax_band"] = pd.Categorical(
            grp["equifax_band"], categories=self._equifax_labels, ordered=True
        )
        grp["deal_band"] = pd.Categorical(
            grp["deal_band"], categories=self._deal_labels, ordered=True
        )

        return grp.sort_values(["equifax_band", "deal_band"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Evidence piece 2: Interaction pattern (bad rate by deal bin per stratum)
    # ------------------------------------------------------------------

    def interaction_pattern(self) -> pd.DataFrame:
        """
        Bad rate by deal variable band, presented separately for each
        Equifax stratum.

        This is the key visual table. When shown as a line chart (one line
        per Equifax band, x-axis = deal band, y-axis = bad rate), the
        differing slopes across strata are the interaction effect that
        Breslow-Day detected.

        If the lines were parallel (same slope), BD would have passed.
        The fact they aren't is why the interaction term is justified.

        Returns
        -------
        pd.DataFrame — wide format: deal_band as index, Equifax bands as columns.
        Values are bad rates. Easier to read in a presentation than long format.
        """
        grid = self.bad_rate_grid()

        wide = grid.pivot(
            index="deal_band",
            columns="equifax_band",
            values="bad_rate",
        )

        # Rename columns to make intent clear when shown to stakeholders
        wide.columns.name = "Equifax Band →"
        wide.index.name   = f"{self.deal_var} Band ↓"

        return wide

    # ------------------------------------------------------------------
    # Evidence piece 2b: Absolute bad count table (for stakeholder trust)
    # ------------------------------------------------------------------

    def bad_count_grid(self) -> pd.DataFrame:
        """
        Same grid as bad_rate_grid() but formatted for a presentation slide:
        each cell shows 'bad_rate (N=n, Bads=b)' as a single string so the
        raw numbers sit alongside the rates.

        Stakeholders can check the raw counts when questioning whether a
        bad rate is driven by small sample size.

        Returns
        -------
        pd.DataFrame — wide format: deal_band as rows, Equifax bands as columns.
        """
        grid = self.bad_rate_grid()

        grid["cell_label"] = (
            (grid["bad_rate"] * 100).round(1).astype(str) + "% "
            + "(N=" + grid["N"].astype(str)
            + ", Bads=" + grid["bads"].astype(str) + ")"
        )

        wide = grid.pivot(
            index="deal_band",
            columns="equifax_band",
            values="cell_label",
        )

        wide.columns.name = "Equifax Band →"
        wide.index.name   = f"{self.deal_var} Band ↓"

        return wide

    # ------------------------------------------------------------------
    # Evidence piece 3: Total score grid
    # ------------------------------------------------------------------

    def total_score_grid(self) -> Optional[pd.DataFrame]:
        """
        Mean total model score by Equifax band × deal band.

        This is the reconciliation step. Individual interaction term
        contributions can appear counterintuitive, but the TOTAL score
        (which includes main effects + interaction terms + intercept)
        should still rank-order customers in a business-sensible direction.

        Returns None if no model was provided at initialisation.

        Returns
        -------
        pd.DataFrame — wide format: deal_band as rows, Equifax bands as columns.
        Values are mean total scores (rounded to 1 decimal place).
        Higher score = lower predicted risk.

        Returns None if model was not provided or scoring failed.
        """
        if self.model is None:
            return None

        grp = (
            self.df
            .groupby(["_equifax_band", "_deal_band"], observed=True)["_score"]
            .mean()
            .round(1)
            .reset_index()
            .rename(columns={
                "_equifax_band": "equifax_band",
                "_deal_band":    "deal_band",
                "_score":        "mean_score",
            })
        )

        wide = grp.pivot(
            index="deal_band",
            columns="equifax_band",
            values="mean_score",
        )

        wide.columns.name = "Equifax Band →"
        wide.index.name   = f"{self.deal_var} Band ↓"

        return wide

    # ------------------------------------------------------------------
    # Evidence piece 4: BD reconciliation text
    # ------------------------------------------------------------------

    def bd_reconciliation(self) -> str:
        """
        Plain-English reconciliation of the Breslow-Day finding with the
        interaction model's point contribution behaviour.

        Intended to be read aloud or included as speaker notes in a
        stakeholder presentation.
        """
        pattern = self.interaction_pattern()

        # Calculate the bad rate reduction from lowest to highest deal band
        # within each Equifax stratum — this is the 'slope' per stratum
        slopes: Dict[str, float] = {}
        for eq_band in self._equifax_labels:
            if eq_band in pattern.columns:
                col = pattern[eq_band].dropna()
                if len(col) >= 2:
                    slopes[eq_band] = float(col.iloc[0] - col.iloc[-1])

        sep      = "=" * 68
        sub_sep  = "-" * 50
        dv       = self.deal_var
        has_score_grid = self.model is not None

        lines = [
            sep,
            "  BRESLOW-DAY → INTERACTION MODEL RECONCILIATION",
            sep,
            "",
            "  WHAT THE BRESLOW-DAY TEST FOUND",
            f"  {sub_sep}",
            f"  Higher {dv} reduces default rate in EVERY Equifax stratum.",
            "  The test found this direction is consistent — there is no stratum",
            "  where higher deposit makes customers worse.",
            "",
            "  However, BD also found the SIZE of that reduction is significantly",
            "  DIFFERENT across strata. That is the interaction.",
            "",
            f"  Bad rate reduction (lowest → highest {dv} band) per stratum:",
        ]

        for band, slope in slopes.items():
            lines.append(
                f"      {band}: {slope*100:+.1f}pp reduction "
                f"from lowest to highest {dv} band"
            )

        lines += [
            "",
            "  WHY THE POINT CONTRIBUTION CAN LOOK COUNTERINTUITIVE",
            f"  {sub_sep}",
            f"  In the interaction model, the effective coefficient on {dv}",
            "  is not a single number — it changes with the customer's Equifax score:",
            "",
            "      Effective slope = β_deal + β_interaction × Equifax_std",
            "",
            "  For a low-Equifax customer (Equifax_std is negative), the interaction",
            "  term subtracts from the main effect. If the interaction coefficient",
            "  is large, this can reduce (or flip) the effective slope for that",
            "  stratum, producing an isolated contribution that looks wrong.",
            "",
        ]

        if has_score_grid:
            lines += [
                "  WHY THE TOTAL SCORE STILL MAKES SENSE",
                f"  {sub_sep}",
                "  The individual term contribution is ONE COMPONENT. When you look",
                "  at the total score grid (Evidence 3 above), the rank ordering",
                "  across Equifax × deal bands should be business-coherent:",
                f"      • High Equifax + High {dv}  → highest total score",
                f"      • Low Equifax  + Low {dv}   → lowest total score",
                "  The interaction term is redistributing score between strata,",
                "  not reversing the overall direction of risk.",
                "",
            ]

        lines += [
            "  SUMMARY FOR STAKEHOLDERS",
            f"  {sub_sep}",
            f"  The model does NOT say high {dv} is bad for any customer.",
            "  It says the DEGREE OF PROTECTION it offers depends on credit quality.",
            "  A high deposit provides strong protection for a good-credit customer.",
            "  For a weaker-credit customer, it provides less protection — the",
            "  Equifax score is already the dominant risk signal in that segment.",
            "",
            "  The interaction term models this difference in magnitude.",
            "  The Breslow-Day test was the statistical confirmation that this",
            "  difference is real and not due to sampling noise.",
            sep,
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plot: interaction pattern
    # ------------------------------------------------------------------

    def plot_interaction_pattern(
        self,
        figsize: Tuple[float, float] = (9, 5),
        title:   Optional[str]       = None,
    ) -> plt.Figure:
        """
        Line chart showing bad rate by deal variable band, one line per
        Equifax stratum.

        Parallel lines → no interaction (homogeneous odds ratios, BD passes).
        Diverging/crossing lines → interaction present (BD fails).

        The slope difference visible in this chart is the core explanation
        for why the point contribution differs across Equifax bands.

        Parameters
        ----------
        figsize : (width, height) in inches.
        title   : optional override for the chart title.

        Returns
        -------
        matplotlib Figure (call fig.savefig(...) to export).
        """
        pattern = self.interaction_pattern()

        fig, ax = plt.subplots(figsize=figsize)

        colours = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

        for i, col in enumerate(pattern.columns):
            vals = pattern[col].dropna()
            ax.plot(
                range(len(vals)),
                vals * 100,
                marker    = "o",
                linewidth = 2.2,
                markersize = 7,
                label     = f"Equifax: {col}",
                color     = colours[i % len(colours)],
            )

        ax.set_xticks(range(len(pattern.index)))
        ax.set_xticklabels(pattern.index.tolist(), fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.1f}%"
        ))

        ax.set_xlabel(f"{self.deal_var} Band", fontsize=11)
        ax.set_ylabel("Bad Rate (%)", fontsize=11)
        ax.set_title(
            title or (
                f"Bad Rate by {self.deal_var} Band per Equifax Stratum\n"
                f"(Parallel lines = no interaction; diverging lines = interaction)"
            ),
            fontsize=12,
            pad=12,
        )
        ax.legend(
            title      = "Equifax Band",
            fontsize   = 9,
            title_fontsize = 9,
            loc        = "upper right",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Plot: bad rate heatmap grid
    # ------------------------------------------------------------------

    def plot_bad_rate_heatmap(
        self,
        figsize: Tuple[float, float] = (8, 5),
        title:   Optional[str]       = None,
    ) -> plt.Figure:
        """
        Heatmap of bad rates by Equifax band × deal variable band.

        Each cell is annotated with bad rate (%) and the raw bad count.
        Darker colour = higher bad rate.

        This is the presentation-ready version of bad_rate_grid() — easier
        for stakeholders to read than a table.

        Parameters
        ----------
        figsize : (width, height) in inches.
        title   : optional override for the chart title.

        Returns
        -------
        matplotlib Figure.
        """
        grid = self.bad_rate_grid()

        # Pivot for heatmap — bad rate values
        pivot_rate = grid.pivot(
            index="deal_band",
            columns="equifax_band",
            values="bad_rate",
        )
        pivot_bads = grid.pivot(
            index="deal_band",
            columns="equifax_band",
            values="bads",
        )
        pivot_n = grid.pivot(
            index="deal_band",
            columns="equifax_band",
            values="N",
        )

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            pivot_rate.values,
            cmap   = "RdYlGn_r",
            aspect = "auto",
            vmin   = 0,
            vmax   = pivot_rate.values.max() * 1.1,
        )

        ax.set_xticks(range(len(pivot_rate.columns)))
        ax.set_xticklabels(pivot_rate.columns.tolist(), fontsize=10)
        ax.set_yticks(range(len(pivot_rate.index)))
        ax.set_yticklabels(pivot_rate.index.tolist(), fontsize=10)

        ax.set_xlabel("Equifax Band →", fontsize=11)
        ax.set_ylabel(f"{self.deal_var} Band ↓", fontsize=11)
        ax.set_title(
            title or f"Bad Rate Grid: {self.deal_var} × Equifax Band",
            fontsize=12,
            pad=10,
        )

        # Annotate each cell with bad rate + raw counts
        for i in range(len(pivot_rate.index)):
            for j in range(len(pivot_rate.columns)):
                rate  = pivot_rate.values[i, j]
                bads  = int(pivot_bads.values[i, j])
                n_val = int(pivot_n.values[i, j])
                if not np.isnan(rate):
                    cell_text = f"{rate*100:.1f}%\n(Bads={bads}, N={n_val})"
                    # Use white text on dark cells, black on light
                    text_colour = "white" if rate > pivot_rate.values.max() * 0.6 else "black"
                    ax.text(
                        j, i,
                        cell_text,
                        ha        = "center",
                        va        = "center",
                        fontsize  = 9,
                        color     = text_colour,
                        fontweight = "bold" if rate > pivot_rate.values.max() * 0.6 else "normal",
                    )

        plt.colorbar(im, ax=ax, label="Bad Rate", format=mticker.FuncFormatter(
            lambda x, _: f"{x:.1%}"
        ))

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Plot: predicted probability vs observed bad rate
    # ------------------------------------------------------------------

    def plot_predicted_vs_actual(
        self,
        n_bands:  int                    = 10,
        figsize:  Tuple[float, float]    = (10, 5),
        title:    Optional[str]          = None,
    ) -> Optional[plt.Figure]:
        """
        Calibration chart — predicted PD vs observed bad rate by score band.

        Customers are sorted by predicted probability and grouped into
        n_bands equal-sized bands. Within each band, the mean predicted
        PD is plotted against the observed default rate.

        A well-calibrated model produces points that sit close to the
        45-degree diagonal. Systematic over- or under-prediction is
        immediately visible.

        Two panels are shown side by side:
            Left  — bars for observed bad rate with predicted PD overlaid
                    as a line. The primary calibration view.
            Right — scatter of predicted vs observed with the 45° line.
                    Makes over/under-prediction easy to see at a glance.

        Requires model and df_model to have been provided at init.

        Parameters
        ----------
        n_bands : number of equal-frequency score bands. Default 10 (deciles).
        figsize : (width, height) in inches.
        title   : optional override for the suptitle.

        Returns
        -------
        matplotlib Figure, or None if the model was not provided / scoring failed.
        """
        if self.model is None or "_proba" not in self.df.columns:
            warnings.warn(
                "plot_predicted_vs_actual requires model and df_model. "
                "Returning None."
            )
            return None

        df = self.df[["_proba", self.target]].dropna().copy()
        df["_band"] = pd.qcut(df["_proba"], q=n_bands, duplicates="drop", labels=False)

        grouped = (
            df.groupby("_band")
            .agg(
                n           =("_proba",     "count"),
                predicted_pd=("_proba",     "mean"),
                observed_br =(self.target,  "mean"),
            )
            .reset_index()
        )
        grouped["band_label"] = (grouped["_band"] + 1).astype(str)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ── Left panel: bar (observed) + line (predicted) ──────────────
        x = np.arange(len(grouped))
        bar_width = 0.6

        ax1.bar(
            x,
            grouped["observed_br"] * 100,
            width     = bar_width,
            color     = "#4C72B0",
            alpha     = 0.75,
            label     = "Observed Bad Rate",
            zorder    = 2,
        )
        ax1.plot(
            x,
            grouped["predicted_pd"] * 100,
            marker    = "o",
            linewidth = 2,
            markersize = 6,
            color     = "#DD4949",
            label     = "Predicted PD",
            zorder    = 3,
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(grouped["band_label"], fontsize=9)
        ax1.set_xlabel("Score Band (1 = lowest PD, higher = riskier)", fontsize=10)
        ax1.set_ylabel("Rate (%)", fontsize=10)
        ax1.set_title("Observed Bad Rate vs Predicted PD\nby Score Band", fontsize=11)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.spines[["top", "right"]].set_visible(False)

        # ── Right panel: scatter with 45° line ─────────────────────────
        ax2.scatter(
            grouped["predicted_pd"] * 100,
            grouped["observed_br"]  * 100,
            s         = grouped["n"] / grouped["n"].max() * 200,
            color     = "#4C72B0",
            alpha     = 0.8,
            zorder    = 3,
            label     = "Score band\n(size ∝ N)",
        )

        # 45° perfect calibration line
        all_vals  = pd.concat([grouped["predicted_pd"], grouped["observed_br"]]) * 100
        line_min  = max(0, all_vals.min() * 0.85)
        line_max  = all_vals.max() * 1.1
        ax2.plot(
            [line_min, line_max],
            [line_min, line_max],
            linestyle = "--",
            color     = "#DD4949",
            linewidth = 1.5,
            label     = "Perfect calibration",
            zorder    = 2,
        )

        ax2.set_xlabel("Predicted PD (%)", fontsize=10)
        ax2.set_ylabel("Observed Bad Rate (%)", fontsize=10)
        ax2.set_title("Predicted vs Observed\n(points on diagonal = perfect calibration)", fontsize=11)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax2.legend(fontsize=9)
        ax2.grid(linestyle="--", alpha=0.4)
        ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            title or f"Model Calibration: {self.deal_var} Interaction Model",
            fontsize=13,
            y=1.01,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Full print report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """
        Print all evidence pieces to stdout in sequence.

        Suitable for a notebook or terminal review before presenting
        to stakeholders. The plots are available separately via:
            plot_interaction_pattern()
            plot_bad_rate_heatmap()
            plot_predicted_vs_actual()
        """
        sep = "=" * 68

        print(f"\n{sep}")
        print(f"  STAKEHOLDER EVIDENCE: {self.deal_var} × {self.equifax_var}")
        print(sep)

        # Evidence 1: bad rate grid
        print("\n  EVIDENCE 1 — Raw Bad Rate Grid  (validates BD finding)")
        print("  " + "-" * 55)
        print("  Read down each Equifax column: does higher deposit reduce")
        print(f"  bad rate within that stratum? It should in all strata.")
        print()
        grid = self.bad_rate_grid()
        print(
            grid[["equifax_band", "deal_band", "N", "bads", "bad_rate", "pct_of_pop"]]
            .rename(columns={
                "equifax_band": "Equifax Band",
                "deal_band":    "Deal Band",
                "bad_rate":     "Bad Rate",
                "pct_of_pop":   "% Pop",
            })
            .to_string(index=False)
        )

        # Evidence 2: formatted cross-tab
        print(f"\n\n  EVIDENCE 2 — Bad Rate by {self.deal_var} × Equifax Band")
        print("  " + "-" * 55)
        print("  Differing slopes across columns = the interaction BD detected.")
        print("  If slopes were equal, BD would have passed.\n")
        print(self.bad_count_grid().to_string())

        # Evidence 3: total score grid (optional)
        score_grid = self.total_score_grid()
        if score_grid is not None:
            print(f"\n\n  EVIDENCE 3 — Mean Total Score Grid (Higher = Lower Risk)")
            print("  " + "-" * 55)
            print("  Individual term contributions can look odd. The TOTAL score")
            print("  should rank-order cells in a business-sensible direction.\n")
            print(score_grid.to_string())
        else:
            print(
                "\n\n  EVIDENCE 3 — Total Score Grid\n"
                "  " + "-" * 55 + "\n"
                "  Skipped — pass model= and df_model= at initialisation to enable."
            )

        # Evidence 4: reconciliation
        print(f"\n\n{self.bd_reconciliation()}")

        # Calibration note
        if self.model is not None and "_proba" in self.df.columns:
            print(
                "\n  Calibration chart available — call "
                "report.plot_predicted_vs_actual() to generate."
            )
