"""
Smoke test for scorecard.comparison.model_comparison.

Generates a synthetic credit-risk dataset with realistic structure,
fits three candidate models that mirror the user's three examples,
and exercises every code path: fit → evaluate → summary → Excel → HTML.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from scorecard.preprocessing.binning import BinningPipeline
from scorecard.comparison import VariableConfig, ModelSpec, ModelComparison


RNG = np.random.default_rng(42)


def make_synthetic_data(n: int) -> pd.DataFrame:
    """
    Synthetic data designed to produce sensible Gini, calibration,
    and stability behaviour for testing.
    """
    equifax = RNG.normal(650, 80, n).clip(300, 900)
    # Deposit value: log-normal, somewhat correlated with Equifax
    deposit_value = np.exp(RNG.normal(7.5, 0.6, n)) + 0.005 * equifax
    # Instalment: also log-normal, weakly inversely correlated with equifax
    instalment = np.exp(RNG.normal(5.5, 0.4, n)) - 0.003 * (equifax - 650)
    instalment = np.maximum(instalment, 50)

    # True log-odds with a real instalment × equifax interaction
    eq_z   = (equifax - 650) / 80
    dep_z  = (np.log(deposit_value) - 7.5) / 0.6
    inst_z = (np.log(instalment) - 5.5) / 0.4

    log_odds = (
        -2.5
        - 0.9 * eq_z
        - 0.3 * dep_z
        + 0.5 * inst_z
        + 0.3 * eq_z * inst_z   # real interaction
        + RNG.normal(0, 0.3, n)
    )
    p = 1 / (1 + np.exp(-log_odds))
    default_flag = (RNG.uniform(0, 1, n) < p).astype(int)

    return pd.DataFrame({
        "equifax_score": equifax,
        "deposit_value": deposit_value,
        "instalment":    instalment,
        "default_flag":  default_flag,
    })


def main() -> None:
    print("Building synthetic dev + OOT samples...")
    dev_df = make_synthetic_data(20_000)
    oot_df = make_synthetic_data(8_000)
    # Small distribution shift on OOT to give PSI/CSI something to detect
    oot_df["instalment"] = oot_df["instalment"] * 1.08
    print(f"  dev:  n={len(dev_df):,}  bad_rate={dev_df['default_flag'].mean():.2%}")
    print(f"  oot:  n={len(oot_df):,}  bad_rate={oot_df['default_flag'].mean():.2%}")

    # Bin variables so the WoE-mode path also has data to consume
    print("\nFitting BinningPipeline (so WoE-mode specs have inputs)...")
    binner = (
        BinningPipeline(target="default_flag")
        .add_variable("deposit_value", n_bins=8)
        .add_variable("instalment",    n_bins=8)
    )
    binner.fit(dev_df)
    dev_df = binner.transform(dev_df)
    oot_df = binner.transform(oot_df)

    # Three candidate models mirroring the user's stated examples
    specs = [
        ModelSpec(
            name           = "M1_Deposit_DepInt_Instalment",
            description    = "Continuous deposit value with interaction; raw instalment",
            deal_variables = ["deposit_value", "instalment"],
            configs        = {
                "deposit_value": VariableConfig(mode="continuous", transform="none"),
                "instalment":    VariableConfig(mode="continuous", transform="none"),
            },
            interactions   = ["deposit_value"],
        ),
        ModelSpec(
            name           = "M2_Deposit_Instalment_InstInt",
            description    = "Continuous deposit value; instalment with interaction",
            deal_variables = ["deposit_value", "instalment"],
            configs        = {
                "deposit_value": VariableConfig(mode="continuous", transform="none"),
                "instalment":    VariableConfig(mode="continuous", transform="none"),
            },
            interactions   = ["instalment"],
        ),
        ModelSpec(
            name           = "M3_SqrtDep_DepInt_LogInst",
            description    = "Sqrt deposit value with interaction; log instalment",
            deal_variables = ["deposit_value", "instalment"],
            configs        = {
                "deposit_value": VariableConfig(mode="continuous", transform="sqrt"),
                "instalment":    VariableConfig(mode="continuous", transform="log"),
            },
            interactions   = ["deposit_value"],
        ),
    ]

    print("\nRunning ModelComparison.fit() → evaluate() → export ...")
    comparison = ModelComparison(
        specs       = specs,
        target      = "default_flag",
        equifax_col = "equifax_score",
    )
    comparison.fit(dev_df)
    comparison.evaluate(dev_df, oot_df)

    print("\nHeadline summary:")
    print(comparison.summary().to_string(index=False))

    out_dir = Path("/home/claude/test_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = comparison.export_excel(
        out_dir / "model_comparison.xlsx",
        scorecard_dev_df=dev_df,
    )
    html_path = comparison.export_html(out_dir / "model_comparison.html")
    print(f"\n  Excel written → {xlsx_path}")
    print(f"  HTML written  → {html_path}")

    # Spot check Excel contents
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    print(f"\n  Excel sheets present: {list(sheets.keys())}")
    print(f"  Row counts per sheet:")
    for name, df in sheets.items():
        print(f"    {name:<22} {len(df):>6,}")

    # Sanity check the HTML is non-trivial
    html_size = html_path.stat().st_size
    print(f"\n  HTML size: {html_size:,} bytes")
    assert html_size > 100_000, "HTML output suspiciously small"
    print("\n✅  All checks passed.")


if __name__ == "__main__":
    main()