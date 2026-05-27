"""
modelling/deal_variable_plots.py

Graphical output for DealVariableLogOddsAnalysis results.

Produces two types of output:

    Per-variable figure (4 panels):
        1. Log-odds per bin — overall population
        2. Log-odds per bin by Equifax band — shows shape divergence
        3. WoE per bin by Equifax band — complements the log-odds view
        4. Transformation comparison — original vs transformed log-odds

    Summary comparison figure (3 panels):
        1. Gini: base model vs interaction model per deal variable
        2. AIC delta — improvement from adding interaction term
        3. LR test p-values — statistical significance of each interaction

Why these charts:
    The log-odds shape chart directly answers whether a transformation is
    needed before WoE binning. If the relationship is curved or non-monotonic,
    the WoE bins are working harder than they should.

    Stratifying by Equifax band shows whether the *shape* changes across
    credit quality or only the level shifts. Shape change = genuine interaction;
    level shift = multiplicative structure approximately holds. This is the
    visual companion to the Breslow-Day test result.

    The WoE-by-band chart adds a second lens: if WoE values for the same
    bin diverge across bands, the bin is encoding different risk in different
    credit tiers. The transformation comparison closes the loop — it shows
    whether the suggested transform actually linearises the relationship.

    The summary comparison figure provides governance-ready evidence for
    why interaction terms were included in the model.

Usage
-----
    from modelling.deal_variable_plots import DealVariablePlotter

    plotter = DealVariablePlotter(save_dir="outputs/plots")

    # After running DealVariableLogOddsAnalysis.run():
    plotter.plot_all_variables(results, df_raw=dev_df, deal_vars=["ltv_ratio"])

    # After running compare_interaction_models():
    plotter.plot_model_comparison(comparison_df)

    # Or via the pipeline:
    pipeline.log_odds_analysis(...)          # generates results internally
    # then retrieve from the returned dict and pass here
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from .deal_variable_analysis import LogOddsResult

# Use non-interactive backend when running without a display
matplotlib.rcParams.update({
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

# Colour palette — one colour per Equifax band, consistent across all charts
_BAND_COLOURS = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]
_BASE_COLOUR   = "#34495e"
_SPARSE_COLOUR = "#e74c3c"
_GOOD_COLOUR   = "#27ae60"


class DealVariablePlotter:
    """
    Produces matplotlib figures from DealVariableLogOddsAnalysis results.

    Parameters
    ----------
    save_dir : directory to save figures (PNG). If None, figures are shown
               interactively (requires a display / notebook environment).
    figsize_per_var : (width, height) for the 4-panel per-variable figure.
    figsize_summary : (width, height) for the summary comparison figure.
    """

    def __init__(
        self,
        save_dir:        Optional[str] = None,
        figsize_per_var: Tuple[float, float] = (16, 12),
        figsize_summary: Tuple[float, float] = (14, 5),
    ) -> None:
        self.save_dir        = Path(save_dir) if save_dir else None
        self.figsize_per_var = figsize_per_var
        self.figsize_summary = figsize_summary

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_all_variables(
        self,
        results:    Dict[str, "LogOddsResult"],   # noqa: F821
        df_raw:     Optional[pd.DataFrame] = None,
        deal_vars:  Optional[List[str]]    = None,
    ) -> List[plt.Figure]:
        """
        Generate one 4-panel figure per deal variable.

        Parameters
        ----------
        results   : output of DealVariableLogOddsAnalysis.run()
        df_raw    : raw (pre-WoE) dataframe; required for transformation
                    comparison panel. If None, that panel is skipped.
        deal_vars : subset of variables to plot. Defaults to all in results.

        Returns
        -------
        List of matplotlib Figure objects (one per variable).
        """
        vars_to_plot = deal_vars or list(results.keys())
        figs = []
        for var in vars_to_plot:
            if var not in results:
                warnings.warn(f"No result found for '{var}' — skipping.")
                continue
            fig = self._plot_variable(var, results[var], df_raw)
            figs.append(fig)
        return figs

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
    ) -> plt.Figure:
        """
        3-panel summary chart comparing base vs interaction model per
        deal variable.

        Expected columns from compare_interaction_models():
            variable, gini_base, gini_interaction, gini_uplift,
            delta_aic, lr_p_value, interaction_significant
        """
        fig = self._build_model_comparison_figure(comparison_df)
        self._save_or_show(fig, "model_comparison_summary")
        return fig

    # ------------------------------------------------------------------
    # Per-variable 4-panel figure
    # ------------------------------------------------------------------

    def _plot_variable(
        self,
        var:    str,
        result: "LogOddsResult",                  # noqa: F821
        df_raw: Optional[pd.DataFrame],
    ) -> plt.Figure:

        fig = plt.figure(figsize=self.figsize_per_var, constrained_layout=True)
        fig.suptitle(
            f"Deal Variable Analysis: {var.replace('_', ' ').title()}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        gs = GridSpec(2, 2, figure=fig)

        ax_lo    = fig.add_subplot(gs[0, 0])   # Panel 1: overall log-odds
        ax_band  = fig.add_subplot(gs[0, 1])   # Panel 2: log-odds by band
        ax_woe   = fig.add_subplot(gs[1, 0])   # Panel 3: WoE by band
        ax_trans = fig.add_subplot(gs[1, 1])   # Panel 4: transformation

        self._panel_log_odds_overall(ax_lo, result)
        self._panel_log_odds_by_band(ax_band, result)
        self._panel_woe_by_band(ax_woe, result)
        self._panel_transformation(ax_trans, var, result, df_raw)

        self._save_or_show(fig, f"logodds_{var}")
        return fig

    # ------------------------------------------------------------------
    # Panel 1: Overall log-odds per bin
    # ------------------------------------------------------------------

    def _panel_log_odds_overall(
        self,
        ax:     plt.Axes,
        result: "LogOddsResult",                  # noqa: F821
    ) -> None:
        df = result.overall_log_odds.copy()
        if df.empty:
            ax.set_title("Overall Log-Odds (no data)")
            return

        x_labels = [str(b) for b in df["bin"]]
        x_pos    = np.arange(len(x_labels))

        # Bar: bad rate (secondary context), line: log-odds (primary)
        ax2 = ax.twinx()
        ax2.bar(
            x_pos, df["bad_rate"],
            color="#bdc3c7", alpha=0.45, label="Bad Rate",
            zorder=1,
        )
        ax2.set_ylabel("Bad Rate", color="#7f8c8d")
        ax2.tick_params(axis="y", labelcolor="#7f8c8d")
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)

        # Colour log-odds markers by sparse flag
        colours = [
            _SPARSE_COLOUR if s else _BASE_COLOUR
            for s in df["sparse"]
        ]
        ax.plot(
            x_pos, df["log_odds"],
            color=_BASE_COLOUR, linewidth=2, zorder=3, label="Log-Odds",
        )
        ax.scatter(x_pos, df["log_odds"], c=colours, s=55, zorder=4)

        # Annotate sparse bins
        for i, (lo, sp) in enumerate(zip(df["log_odds"], df["sparse"])):
            if sp:
                ax.annotate(
                    "sparse", (x_pos[i], lo),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=6.5, color=_SPARSE_COLOUR, ha="center",
                )

        ax.axhline(0, color="#95a5a6", linewidth=0.8, linestyle=":")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=7)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Log-Odds")
        ax.set_title(
            f"Overall Log-Odds per Bin\n"
            f"Shape: {result.linearity_flag}  |  "
            f"Spearman ρ = {result.spearman_overall:.3f}"
            if not np.isnan(result.spearman_overall) else
            f"Overall Log-Odds per Bin\nShape: {result.linearity_flag}"
        )
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=7)

    # ------------------------------------------------------------------
    # Panel 2: Log-odds by Equifax band
    # ------------------------------------------------------------------

    def _panel_log_odds_by_band(
        self,
        ax:     plt.Axes,
        result: "LogOddsResult",                  # noqa: F821
    ) -> None:
        df = result.band_log_odds
        if df.empty:
            ax.set_title("Log-Odds by Equifax Band (no data)")
            return

        bands    = df["equifax_band"].unique().tolist()
        colours  = _band_colours(len(bands))

        # All bins in order (from overall)
        bin_order = result.overall_log_odds["bin"].tolist()
        x_labels  = [str(b) for b in bin_order]
        x_pos     = np.arange(len(x_labels))

        for band, colour in zip(bands, colours):
            band_df = df[df["equifax_band"] == band].copy()
            # Align to the common bin order
            band_df = band_df.set_index("bin").reindex(bin_order)
            lo_vals = band_df["log_odds"].values

            ax.plot(
                x_pos, lo_vals,
                color=colour, linewidth=1.8,
                marker="o", markersize=4,
                label=str(band),
            )

        ax.axhline(0, color="#95a5a6", linewidth=0.8, linestyle=":")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=7)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Log-Odds")
        ax.set_title(
            "Log-Odds per Bin by Equifax Band\n"
            "Parallel lines → level shift only; diverging → genuine interaction"
        )
        ax.legend(title="Equifax Band", loc="best")

        # Shade the per-bin log-odds range to make divergence visible
        lo_wide = df.pivot_table(
            index="bin", columns="equifax_band", values="log_odds"
        ).reindex(bin_order)
        if not lo_wide.empty:
            lo_min = lo_wide.min(axis=1).values
            lo_max = lo_wide.max(axis=1).values
            ax.fill_between(
                x_pos, lo_min, lo_max,
                alpha=0.10, color="#7f8c8d", label="Band range",
            )

    # ------------------------------------------------------------------
    # Panel 3: WoE by Equifax band
    # ------------------------------------------------------------------

    def _panel_woe_by_band(
        self,
        ax:     plt.Axes,
        result: "LogOddsResult",                  # noqa: F821
    ) -> None:
        df = result.band_woe
        if df.empty:
            ax.set_title("WoE by Equifax Band (no data)")
            return

        bands    = df["equifax_band"].unique().tolist()
        colours  = _band_colours(len(bands))
        bin_order = result.overall_log_odds["bin"].tolist()
        x_labels  = [str(b) for b in bin_order]
        x_pos     = np.arange(len(x_labels))

        for band, colour in zip(bands, colours):
            band_df = (
                df[df["equifax_band"] == band]
                .set_index("bin")
                .reindex(bin_order)
            )
            ax.plot(
                x_pos, band_df["woe"].values,
                color=colour, linewidth=1.8,
                marker="s", markersize=4,
                label=str(band),
            )

        ax.axhline(0, color="#95a5a6", linewidth=0.8, linestyle=":")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=7)
        ax.set_xlabel("Bin")
        ax.set_ylabel("WoE")
        ax.set_title(
            "WoE per Bin by Equifax Band\n"
            "Consistent WoE → stable risk signal; diverging → band-specific effect"
        )
        ax.legend(title="Equifax Band", loc="best")

    # ------------------------------------------------------------------
    # Panel 4: Transformation comparison
    # ------------------------------------------------------------------

    def _panel_transformation(
        self,
        ax:     plt.Axes,
        var:    str,
        result: "LogOddsResult",                  # noqa: F821
        df_raw: Optional[pd.DataFrame],
    ) -> None:
        """
        Plot original log-odds alongside post-transformation log-odds.

        If transform_type is "none" or "na", shows a clean message instead
        of a redundant duplicate. If df_raw is not provided, shows a
        placeholder asking the caller to pass the raw dataframe.
        """
        transform_type = getattr(result, "transform_type", "none")

        if transform_type in ("none", "na"):
            ax.set_axis_off()
            msg = (
                "No transformation required.\n"
                "Log-odds relationship is sufficiently linear\n"
                "for WoE binning to capture it well."
                if transform_type == "none" else
                "Transformation not applicable\n(categorical variable)."
            )
            ax.text(
                0.5, 0.5, msg,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9, color="#7f8c8d",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.6", fc="#ecf0f1", ec="#bdc3c7"),
            )
            ax.set_title("Transformation Comparison")
            return

        if df_raw is None or var not in df_raw.columns:
            ax.set_axis_off()
            ax.text(
                0.5, 0.5,
                f"Pass df_raw= to plot_all_variables()\nto see the {transform_type} "
                "transformation comparison.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#7f8c8d", style="italic",
                bbox=dict(boxstyle="round,pad=0.6", fc="#ecf0f1", ec="#bdc3c7"),
            )
            ax.set_title("Transformation Comparison")
            return

        target_col = _find_target_col(df_raw, result)
        if target_col is None:
            ax.set_axis_off()
            ax.set_title("Transformation Comparison (target column not found)")
            return

        # Compute transformed log-odds
        raw_vals = df_raw[var].dropna()
        try:
            trans_vals, trans_label = _apply_transform(
                df_raw[var].values, transform_type
            )
        except Exception as e:
            ax.set_axis_off()
            ax.set_title(f"Transformation failed: {e}")
            return

        df_trans = df_raw[[var, target_col]].copy()
        df_trans["_trans"] = trans_vals

        try:
            df_trans["_trans_bin"] = pd.qcut(
                df_trans["_trans"], q=10, duplicates="drop"
            )
        except Exception:
            df_trans["_trans_bin"] = pd.cut(
                df_trans["_trans"], bins=5
            )

        lo_trans = _compute_simple_log_odds(df_trans, "_trans_bin", target_col)

        # Original log-odds (from result)
        orig = result.overall_log_odds.copy()
        orig_x = np.linspace(0, 1, len(orig))
        trans_x = np.linspace(0, 1, len(lo_trans))

        ax.plot(
            orig_x, orig["log_odds"].values,
            color=_BASE_COLOUR, linewidth=2,
            marker="o", markersize=4,
            label="Original",
        )
        ax.plot(
            trans_x, lo_trans["log_odds"].values,
            color=_GOOD_COLOUR, linewidth=2,
            marker="s", markersize=4, linestyle="--",
            label=f"After {trans_label}",
        )

        ax.axhline(0, color="#95a5a6", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Normalised bin position (0 = lowest, 1 = highest)")
        ax.set_ylabel("Log-Odds")
        ax.set_title(
            f"Transformation Comparison: Original vs {trans_label}\n"
            f"Suggested: {transform_type.upper()}  |  "
            f"A smoother, more linear curve confirms the transformation helps."
        )
        ax.legend(loc="best")

    # ------------------------------------------------------------------
    # Summary comparison figure
    # ------------------------------------------------------------------

    def _build_model_comparison_figure(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        """
        3-panel figure: Gini comparison, AIC delta, LR p-values.
        """
        required = {"variable", "gini_base", "gini_interaction",
                    "delta_aic", "lr_p_value", "interaction_significant"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"comparison_df is missing columns: {missing}. "
                "Pass the output of compare_interaction_models()."
            )

        variables = df["variable"].tolist()
        x_pos     = np.arange(len(variables))
        bar_w     = 0.35

        fig, axes = plt.subplots(1, 3, figsize=self.figsize_summary,
                                 constrained_layout=True)
        fig.suptitle(
            "With vs Without Interaction — Model Comparison per Deal Variable",
            fontsize=11, fontweight="bold",
        )

        # --- Panel 1: Gini ---
        ax = axes[0]
        ax.bar(x_pos - bar_w / 2, df["gini_base"],
               width=bar_w, label="Base (no interaction)",
               color="#7f8c8d", alpha=0.85)
        ax.bar(x_pos + bar_w / 2, df["gini_interaction"],
               width=bar_w, label="With interaction",
               color="#2980b9", alpha=0.85)

        for i, (gb, gi) in enumerate(
            zip(df["gini_base"], df["gini_interaction"])
        ):
            uplift = gi - gb
            colour = _GOOD_COLOUR if uplift >= 0 else _SPARSE_COLOUR
            ax.annotate(
                f"{uplift:+.3f}",
                (x_pos[i] + bar_w / 2, gi),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=7, color=colour, fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables, rotation=30, ha="right")
        ax.set_ylabel("Gini Coefficient")
        ax.set_title("Gini: Base vs Interaction\n(annotation = Gini uplift)")
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # --- Panel 2: ΔAIC ---
        ax = axes[1]
        colours_aic = [
            _GOOD_COLOUR if v < 0 else _SPARSE_COLOUR
            for v in df["delta_aic"]
        ]
        bars = ax.bar(x_pos, df["delta_aic"], color=colours_aic, alpha=0.85)
        ax.axhline(0, color="#2c3e50", linewidth=1.0)
        ax.axhline(-2, color="#27ae60", linewidth=0.8,
                   linestyle="--", label="Δ AIC = -2 threshold")

        for bar, val in zip(bars, df["delta_aic"]):
            ypos = val - 0.5 if val < 0 else val + 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f}", ha="center", va="top" if val < 0 else "bottom",
                fontsize=7, fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables, rotation=30, ha="right")
        ax.set_ylabel("Δ AIC  (interaction − base)")
        ax.set_title("AIC Change from Adding Interaction\n(green = improvement)")
        ax.legend(fontsize=7)

        # --- Panel 3: LR p-values ---
        ax = axes[2]
        colours_lr = [
            _GOOD_COLOUR if sig else "#95a5a6"
            for sig in df["interaction_significant"]
        ]
        ax.bar(x_pos, df["lr_p_value"], color=colours_lr, alpha=0.85)
        ax.axhline(0.05, color=_SPARSE_COLOUR, linewidth=1.2,
                   linestyle="--", label="p = 0.05")

        for i, p in enumerate(df["lr_p_value"]):
            ax.text(
                x_pos[i], p + 0.005, f"{p:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables, rotation=30, ha="right")
        ax.set_ylabel("LR Test p-value  (1 df)")
        ax.set_title(
            "Likelihood Ratio Test p-value\n"
            "(green = significant at 5%;  lower = stronger evidence)"
        )
        ax.legend(fontsize=7)

        return fig

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _save_or_show(self, fig: plt.Figure, name: str) -> None:
        if self.save_dir:
            path = self.save_dir / f"{name}.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"  Saved: {path}")
            plt.close(fig)
        else:
            plt.show()


    # ------------------------------------------------------------------
    # Actual vs Predicted log-odds by Equifax band
    # ------------------------------------------------------------------
 
    def plot_actual_vs_predicted(
        self,
        var:          str,
        result:       "LogOddsResult",                   # noqa: F821
        model:        "InteractionLogisticRegression",   # noqa: F821
        df:           pd.DataFrame,
        equifax_col:  str,
        band_labels:  List[str],
        model_label:  Optional[str]    = None,
        cut_points:   Optional[List[float]] = None,
        n_bins:       int              = 10,
    ) -> plt.Figure:
        """
        Compare empirical log-odds (actual) against model-predicted log-odds
        for each deal variable bin, split by Equifax band.
 
        Each subplot covers one Equifax band. Within each subplot:
            - Solid line  : actual (empirical) log-odds per bin
            - Dashed line : model predicted log-odds per bin
            - Shaded area : gap between actual and predicted
            - RMSE annotated in the subtitle
 
        A well-fitted model will have predicted lines tracking closely
        to actual lines across all bands. Systematic gaps in specific
        bands indicate where the model under- or over-predicts by
        credit quality tier.
 
        Parameters
        ----------
        var          : deal variable name
        result       : output of DealVariableLogOddsAnalysis.run()[var]
        model        : fitted InteractionLogisticRegression
        df           : dataframe used for fitting (must contain equifax_col
                       and all columns required by the model)
        equifax_col  : raw Equifax score column
        band_labels  : Equifax band labels ordered low → high credit quality
        model_label  : display name for the model in legend/title
        cut_points   : optional explicit cut points for the deal variable.
                       If None, bin edges are inferred from result.
        n_bins       : fallback bin count if edges cannot be inferred
        """
        label     = model_label or getattr(model, "model_name", "Model")
        pred_df   = self._predicted_log_odds_by_band(
            df, var, model, result, equifax_col, band_labels, cut_points, n_bins
        )
        return self._draw_actual_vs_predicted(
            var, result, [(label, pred_df, "#2980b9")],
            band_labels, title_prefix=f"Actual vs Predicted — {label}",
            save_name=f"avp_{var}_{label.replace(' ', '_')}",
        )
 
    def plot_model_comparison_bands(
        self,
        var:          str,
        result:       "LogOddsResult",                          # noqa: F821
        models:       "List[InteractionLogisticRegression]",    # noqa: F821
        df:           pd.DataFrame,
        equifax_col:  str,
        band_labels:  List[str],
        model_labels: Optional[List[str]] = None,
        cut_points:   Optional[List[float]] = None,
        n_bins:       int = 10,
    ) -> plt.Figure:
        """
        Overlay predicted log-odds from multiple models against the actual
        log-odds, split by Equifax band.
 
        Use this to compare:
            - Full model vs interaction-only vs main-only
            - Different variable combination models
            - Any set of fitted InteractionLogisticRegression models
 
        The actual log-odds is plotted as a thick grey line. Each model's
        predictions appear in a distinct colour and line style. The model
        whose predictions most closely track the actual line is the best fit
        in log-odds space for that band.
 
        Parameters
        ----------
        var          : deal variable name
        result       : output of DealVariableLogOddsAnalysis.run()[var]
        models       : list of fitted InteractionLogisticRegression models
        df           : dataframe (must satisfy all models' column requirements)
        equifax_col  : raw Equifax score column
        band_labels  : Equifax band labels ordered low → high
        model_labels : display names for each model. Defaults to model_name.
        cut_points   : optional explicit cut points for the deal variable
        n_bins       : fallback bin count if edges cannot be inferred
        """
        _PRED_COLOURS  = ["#2980b9", "#e67e22", "#8e44ad", "#16a085", "#c0392b"]
        _PRED_STYLES   = ["--", "-.", ":", "--", "-."]
 
        labels  = model_labels or [getattr(m, "model_name", f"Model {i+1}")
                                   for i, m in enumerate(models)]
        series  = []
        for m, lbl, col, sty in zip(
            models, labels,
            _PRED_COLOURS[:len(models)],
            _PRED_STYLES[:len(models)],
        ):
            pred_df = self._predicted_log_odds_by_band(
                df, var, m, result, equifax_col, band_labels, cut_points, n_bins
            )
            series.append((lbl, pred_df, col, sty))
 
        return self._draw_actual_vs_predicted(
            var, result,
            [(lbl, pdf, col) for lbl, pdf, col, _ in series],
            band_labels,
            linestyles=[sty for _, _, _, sty in series],
            title_prefix=f"Model Comparison — {var.replace('_', ' ').title()}",
            save_name=f"model_cmp_{var}",
        )
 
    def plot_transform_comparison_bands(
        self,
        var:         str,
        result:      "LogOddsResult",          # noqa: F821
        df:          pd.DataFrame,
        target_col:  str,
        equifax_col: str,
        band_labels: List[str],
        transforms:  Optional[List[str]] = None,
        n_bins:      int = 10,
    ) -> plt.Figure:
        """
        Fit a simple interaction model for each transform variant and compare
        how closely each variant's predicted log-odds tracks the actual
        log-odds, per Equifax band.
 
        Transforms tested:
            "none"  → raw variable, standardised
            "log"   → log(x + shift), standardised
            "sqrt"  → sqrt(x + shift), standardised
            "poly2" → x and x² both standardised and entered as separate
                      main effects and interactions (polynomial regression)
 
        Each subplot (one per Equifax band) shows:
            - Actual log-odds (thick solid grey)
            - Predicted log-odds per transform (coloured dashed lines)
            - RMSE per transform annotated in the legend
 
        This chart directly answers the question: which transformation of
        this deal variable produces predictions closest to the observed
        log-odds pattern, and does it hold consistently across credit quality
        bands?
 
        Parameters
        ----------
        var         : deal variable name (raw column in df)
        result      : output of DealVariableLogOddsAnalysis.run()[var]
        df          : raw dataframe (must contain var, equifax_col, target_col)
        target_col  : binary target column
        equifax_col : raw Equifax score column
        band_labels : Equifax band labels ordered low → high
        transforms  : list of transforms to compare. Defaults to all four.
        n_bins      : bin count for deal variable binning
        """
        from sklearn.preprocessing import StandardScaler as _SS
 
        transforms = transforms or ["none", "log", "sqrt", "poly2"]
 
        _TRANSFORM_COLOURS = {
            "none":  "#7f8c8d",
            "log":   "#2980b9",
            "sqrt":  "#27ae60",
            "poly2": "#e67e22",
        }
        _TRANSFORM_STYLES  = {
            "none":  "--",
            "log":   "-.",
            "sqrt":  ":",
            "poly2": (0, (5, 1)),
        }
 
        series = []
        for transform in transforms:
            pred_log_odds, label = _fit_transform_model(
                df, var, target_col, equifax_col, transform
            )
            if pred_log_odds is None:
                warnings.warn(f"Transform '{transform}' failed — skipping.")
                continue
 
            df_pred = df.copy()
            df_pred["_pred_log_odds"] = pred_log_odds
            df_pred["_deal_bin"]      = self._bin_variable_consistently(
                df, var, result, n_bins
            )
            df_pred["_eq_band"]       = self._assign_equifax_bands(
                df, equifax_col, band_labels
            )
 
            grouped = (
                df_pred.groupby(["_eq_band", "_deal_bin"], observed=True)
                ["_pred_log_odds"].mean()
                .reset_index()
            )
            grouped.columns = ["equifax_band", "bin", "pred_log_odds"]
 
            series.append((
                label,
                grouped,
                _TRANSFORM_COLOURS.get(transform, "#2c3e50"),
                _TRANSFORM_STYLES.get(transform, "--"),
            ))
 
        return self._draw_actual_vs_predicted(
            var, result,
            [(lbl, pdf, col) for lbl, pdf, col, _ in series],
            band_labels,
            linestyles=[sty for _, _, _, sty in series],
            title_prefix=f"Transform Comparison — {var.replace('_', ' ').title()}",
            save_name=f"transform_cmp_{var}",
            show_rmse=True,
        )
 
    # ------------------------------------------------------------------
    # Core drawing engine shared by all three public methods above
    # ------------------------------------------------------------------
 
    def _draw_actual_vs_predicted(
        self,
        var:           str,
        result:        "LogOddsResult",               # noqa: F821
        pred_series:   List[Tuple],                   # [(label, df, colour), ...]
        band_labels:   List[str],
        linestyles:    Optional[List] = None,
        title_prefix:  str = "",
        save_name:     str = "avp",
        show_rmse:     bool = True,
    ) -> plt.Figure:
        """
        Shared rendering engine. Produces a grid of subplots (one per band),
        each showing the actual log-odds and one or more predicted series.
        """
        n_bands   = len(band_labels)
        n_cols    = min(n_bands, 4)
        n_rows    = (n_bands + n_cols - 1) // n_cols
        fig_w     = 5.5 * n_cols
        fig_h     = 4.5 * n_rows
 
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            constrained_layout=True,
        )
        axes_flat = np.array(axes).flatten() if n_bands > 1 else [axes]
 
        fig.suptitle(title_prefix, fontsize=11, fontweight="bold")
 
        bin_order   = result.overall_log_odds["bin"].tolist()
        x_labels    = [str(b) for b in bin_order]
        x_pos       = np.arange(len(x_labels))
 
        linestyles = linestyles or ["--"] * len(pred_series)
 
        for ax_idx, band in enumerate(band_labels):
            ax = axes_flat[ax_idx]
 
            # --- Actual log-odds ---
            actual_df = (
                result.band_log_odds[result.band_log_odds["equifax_band"] == band]
                .set_index("bin")
                .reindex(bin_order)
            )
            actual_lo = actual_df["log_odds"].values
            sparse    = actual_df["sparse"].values if "sparse" in actual_df.columns                         else np.zeros(len(actual_lo), dtype=bool)
 
            ax.plot(
                x_pos, actual_lo,
                colour="#2c3e50", linewidth=2.5,
                marker="o", markersize=5,
                label="Actual", zorder=4,
            )
            # Highlight sparse actual bins
            for i, (lo, sp) in enumerate(zip(actual_lo, sparse)):
                if sp and not np.isnan(lo):
                    ax.scatter(i, lo, colour=_SPARSE_COLOUR,
                               s=60, zorder=5, marker="x")
 
            # --- Predicted series ---
            for (lbl, pred_df, colour), lstyle in zip(pred_series, linestyles):
                band_pred = (
                    pred_df[pred_df["equifax_band"] == band]
                    .set_index("bin")
                    .reindex(bin_order)
                )
                if "pred_log_odds" not in band_pred.columns or band_pred.empty:
                    continue
                pred_lo = band_pred["pred_log_odds"].values
 
                # RMSE vs actual for this band and series
                rmse_val = np.nan
                mask = ~(np.isnan(actual_lo) | np.isnan(pred_lo))
                if mask.sum() > 1:
                    rmse_val = float(np.sqrt(np.mean((actual_lo[mask] - pred_lo[mask]) ** 2)))
 
                legend_lbl = (
                    f"{lbl}  RMSE={rmse_val:.3f}" if show_rmse and not np.isnan(rmse_val)
                    else lbl
                )
                ax.plot(
                    x_pos, pred_lo,
                    colour=colour, linewidth=1.8,
                    linestyle=lstyle, marker="s", markersize=4,
                    label=legend_lbl, zorder=3,
                )
 
                # Shade the gap between actual and predicted
                if len(pred_series) == 1 and not np.all(np.isnan(actual_lo)):
                    ax.fill_between(
                        x_pos, actual_lo, pred_lo,
                        where=~(np.isnan(actual_lo) | np.isnan(pred_lo)),
                        alpha=0.12, colour=colour,
                        label="_nolegend_",
                    )
 
            ax.axhline(0, colour="#95a5a6", linewidth=0.7, linestyle=":")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("Log-Odds")
            ax.set_title(band, fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, loc="best")
 
        # Hide any unused subplots
        for ax in axes_flat[n_bands:]:
            ax.set_visible(False)
 
        self._save_or_show(fig, save_name)
        return fig
 
    # ------------------------------------------------------------------
    # Shared private helpers for actual vs predicted methods
    # ------------------------------------------------------------------
 
    def _predicted_log_odds_by_band(
        self,
        df:          pd.DataFrame,
        var:         str,
        model:       "InteractionLogisticRegression",  # noqa: F821
        result:      "LogOddsResult",                  # noqa: F821
        equifax_col: str,
        band_labels: List[str],
        cut_points:  Optional[List[float]],
        n_bins:      int,
    ) -> pd.DataFrame:
        """
        Compute mean model-predicted log-odds per deal variable bin × Equifax band.
 
        Returns DataFrame: equifax_band | bin | pred_log_odds | n
        """
        df = df.copy()
 
        # Model predictions → log-odds
        y_pred = model.predict_proba(df)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        df["_pred_lo"] = np.log(y_pred / (1 - y_pred))
 
        # Equifax bands
        df["_eq_band"] = self._assign_equifax_bands(df, equifax_col, band_labels)
 
        # Deal variable bins (consistent with LogOddsResult)
        df["_deal_bin"] = self._bin_variable_consistently(
            df, var, result, n_bins, cut_points
        )
 
        grouped = (
            df.groupby(["_eq_band", "_deal_bin"], observed=True)["_pred_lo"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped.columns = ["equifax_band", "bin", "pred_log_odds", "n"]
        return grouped
 
    @staticmethod
    def _assign_equifax_bands(
        df:          pd.DataFrame,
        equifax_col: str,
        band_labels: List[str],
    ) -> pd.Series:
        """
        Assign equal-frequency Equifax bands to each row.
        Matches the logic in DealVariableLogOddsAnalysis.run().
        """
        return pd.qcut(
            df[equifax_col],
            q      = len(band_labels),
            labels = band_labels,
            duplicates = "drop",
        )
 
    @staticmethod
    def _bin_variable_consistently(
        df:          pd.DataFrame,
        var:         str,
        result:      "LogOddsResult",          # noqa: F821
        n_bins:      int = 10,
        cut_points:  Optional[List[float]] = None,
    ) -> pd.Series:
        """
        Assign deal variable bins that match those stored in result.overall_log_odds.
 
        Priority:
        1. Use pd.IntervalIndex from result bins (most accurate — preserves exact edges)
        2. Use explicit cut_points if provided
        3. Fall back to equal-frequency qcut with n_bins
        """
        bins_series = result.overall_log_odds["bin"]
 
        if bins_series.empty:
            return pd.Series(np.nan, index=df.index)
 
        first_bin = bins_series.iloc[0]
 
        # Categorical variable — map to string categories
        if not isinstance(first_bin, pd.Interval):
            return df[var].fillna("Missing").astype(str)
 
        # Try to use the exact IntervalIndex from the result
        try:
            interval_index = pd.IntervalIndex(bins_series.values)
            return pd.cut(df[var], bins=interval_index)
        except Exception:
            pass
 
        # Fall back to explicit cut points
        if cut_points:
            edges = [-np.inf] + sorted(cut_points) + [np.inf]
            return pd.cut(df[var], bins=edges)
 
        # Last resort: equal-frequency bins
        try:
            return pd.qcut(df[var], q=n_bins, duplicates="drop")
        except Exception:
            return pd.Series(np.nan, index=df.index)
 
    def _save_or_show(self, fig: plt.Figure, name: str) -> None:
        if self.save_dir:
            path = self.save_dir / f"{name}.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"  Saved: {path}")
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fit_transform_model(
    df:          pd.DataFrame,
    var:         str,
    target_col:  str,
    equifax_col: str,
    transform:   str,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Fit a simple Equifax × deal variable interaction logistic regression
    using the named transform, and return predicted log-odds for each row.
 
    Models fitted:
        "none"  : logit(PD) = β0 + β_eq·Eq_std + β_j·x_std + β_ij·(Eq_std×x_std)
        "log"   : same but x_std = std(log(x + shift))
        "sqrt"  : same but x_std = std(sqrt(x + shift))
        "poly2" : logit(PD) = β0 + β_eq·Eq_std
                                  + β_1·x_std + β_2·x²_std
                                  + β_int1·(Eq_std×x_std)
                                  + β_int2·(Eq_std×x²_std)
 
    Returns
    -------
    (predicted_log_odds, human_readable_label)
    Returns (None, label) if fitting fails.
    """
    try:
        import statsmodels.api as sm
        from sklearn.preprocessing import StandardScaler as _SS
 
        y        = df[target_col].values
        raw_vals = df[var].values.astype(float)
        eq_raw   = df[equifax_col].values.astype(float)
 
        # Standardise Equifax
        eq_std = _SS().fit_transform(eq_raw.reshape(-1, 1)).flatten()
 
        # Apply transform to deal variable
        if transform == "none":
            x_vals, label = raw_vals, f"{var} (none)"
        elif transform == "log":
            shift  = max(0.0, -raw_vals.min()) + 1e-6
            x_vals = np.log(raw_vals + shift)
            label  = f"log({var})"
        elif transform == "sqrt":
            shift  = max(0.0, -raw_vals.min())
            x_vals = np.sqrt(raw_vals + shift)
            label  = f"sqrt({var})"
        elif transform == "poly2":
            # poly2 handled separately below
            label  = f"{var}²  (polynomial)"
        else:
            return None, transform
 
        if transform == "poly2":
            # Standardise x and x²
            x_std  = _SS().fit_transform(raw_vals.reshape(-1, 1)).flatten()
            x2_std = _SS().fit_transform((raw_vals ** 2).reshape(-1, 1)).flatten()
 
            X = pd.DataFrame({
                "eq":       eq_std,
                "x":        x_std,
                "x2":       x2_std,
                "eq_x":     eq_std * x_std,
                "eq_x2":    eq_std * x2_std,
            })
        else:
            x_std = _SS().fit_transform(x_vals.reshape(-1, 1)).flatten()
            X = pd.DataFrame({
                "eq":    eq_std,
                "x":     x_std,
                "eq_x":  eq_std * x_std,
            })
 
        X_const = sm.add_constant(X)
        fit     = sm.Logit(y, X_const).fit(disp=0)
        y_pred  = fit.predict(X_const).values
        y_pred  = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.log(y_pred / (1 - y_pred)), label
 
    except Exception as e:
        warnings.warn(f"Transform model '{transform}' failed: {e}")
        return None, transform

        
def _band_colours(n: int) -> List[str]:
    """Return n colours from the band palette, cycling if needed."""
    palette = _BAND_COLOURS + [
        "#8e44ad", "#16a085", "#d35400", "#2c3e50",
    ]
    return [palette[i % len(palette)] for i in range(n)]


def _apply_transform(
    values: np.ndarray,
    transform_type: str,
) -> Tuple[np.ndarray, str]:
    """
    Apply the named transformation to a numpy array.

    Returns (transformed_values, human_readable_label).
    Handles negative and zero values safely.
    """
    if transform_type == "log":
        shift = max(0, -values.min()) + 1e-6
        return np.log(values + shift), f"log(x + {shift:.3g})"
    elif transform_type == "sqrt":
        shift = max(0, -values.min())
        return np.sqrt(values + shift), f"sqrt(x + {shift:.3g})" if shift > 0 else "sqrt(x)"
    elif transform_type == "split":
        # For split, create a "distance from median" feature as a proxy
        med = float(np.nanmedian(values))
        return np.abs(values - med), f"|x − median| (split at {med:.2f})"
    else:
        return values, "none"


def _compute_simple_log_odds(
    df:         pd.DataFrame,
    bin_col:    str,
    target_col: str,
) -> pd.DataFrame:
    """Compute empirical log-odds per bin for a binned series."""
    rows = []
    for bin_val, grp in df.groupby(bin_col, observed=True):
        n_total = len(grp)
        n_bads  = int(grp[target_col].sum())
        n_goods = n_total - n_bads
        n_bads_adj  = max(n_bads,  0.5)
        n_goods_adj = max(n_goods, 0.5)
        rows.append({
            "bin":      bin_val,
            "n_total":  n_total,
            "n_bads":   n_bads,
            "log_odds": float(np.log(n_bads_adj / n_goods_adj)),
        })
    return pd.DataFrame(rows)


def _find_target_col(
    df_raw: pd.DataFrame,
    result: "LogOddsResult",                       # noqa: F821
) -> Optional[str]:
    """
    Try to locate the binary target column in df_raw.
    Checks common names; falls back to the first binary int column.
    """
    candidates = ["default_flag", "default", "bad", "target", "outcome"]
    for c in candidates:
        if c in df_raw.columns:
            return c
    # Fallback: first column with only 0/1 values
    for col in df_raw.columns:
        if df_raw[col].dropna().isin([0, 1]).all():
            return col
    return None
