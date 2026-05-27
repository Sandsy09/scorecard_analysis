import unittest
from types import SimpleNamespace

from modelling.model_comparison import ModelComparison


def _fake_model(name, terms, llf):
    return SimpleNamespace(
        model_name=name,
        deal_variables=["ltv_ratio"],
        _design_columns=terms,
        diagnostics={
            "log_likelihood": llf,
            "n_params": len(terms) + 1,
            "aic": 0.0,
            "bic": 0.0,
        },
    )


class ModelComparisonTests(unittest.TestCase):
    def test_lr_tests_use_design_terms_not_only_deal_variable_subsets(self):
        main_only = _fake_model(
            "main_only",
            ["equifax_std", "cont_ltv_ratio"],
            -10.0,
        )
        interaction_only = _fake_model(
            "interaction_only",
            ["equifax_std", "equifax_x_ltv_ratio"],
            -9.5,
        )
        full = _fake_model(
            "full",
            ["equifax_std", "cont_ltv_ratio", "equifax_x_ltv_ratio"],
            -8.0,
        )

        lr_tests = ModelComparison([main_only, interaction_only, full]).pairwise_lr_tests()

        pairs = {
            (row["restricted"], row["full"], row["added_terms"])
            for _, row in lr_tests.iterrows()
        }

        self.assertIn(("main_only", "full", "equifax_x_ltv_ratio"), pairs)
        self.assertIn(("interaction_only", "full", "cont_ltv_ratio"), pairs)


if __name__ == "__main__":
    unittest.main()

