import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pandas as pd

from interaction_pipeline import InteractionScorecardPipeline
from modelling.interaction_model import DealVariableConfig


class _FakeBinner:
    def get_selected_variables(self, min_iv=0.10, max_iv=0.50):
        return ["ltv_ratio"]


class InteractionPipelineConfigTests(unittest.TestCase):
    def test_fit_all_term_structures_preserves_base_mode_and_transform(self):
        created_models = []

        class FakeInteractionModel:
            def __init__(
                self,
                equifax_col,
                deal_variables,
                target,
                model_name="",
                deal_configs=None,
                include_equifax_main=True,
            ):
                self.equifax_col = equifax_col
                self.deal_variables = deal_variables
                self.target = target
                self.model_name = model_name
                self.deal_configs = deal_configs or {}
                self.include_equifax_main = include_equifax_main
                self.diagnostics = {
                    "aic": 1.0,
                    "bic": 1.0,
                    "log_likelihood": -1.0,
                    "n_params": 1,
                }
                created_models.append(self)

            def fit(self, df):
                return self

            def diagnostic_report(self):
                return "diagnostics"

        pipe = InteractionScorecardPipeline(
            target="default_flag",
            equifax_col="equifax_score",
            deal_variables=["ltv_ratio"],
        )
        pipe.dev_woe = pd.DataFrame({
            "equifax_score": [600],
            "ltv_ratio": [80],
            "default_flag": [0],
        })
        pipe.deal_binner = _FakeBinner()

        configs = {
            "ltv_ratio": DealVariableConfig(
                mode="continuous",
                transform="log",
            )
        }

        with (
            patch("interaction_pipeline.InteractionLogisticRegression", FakeInteractionModel),
            redirect_stdout(StringIO()),
        ):
            pipe.fit_all_term_structures(
                max_combo_size=1,
                min_iv=0.0,
                max_iv=1.0,
                deal_configs=configs,
            )

        self.assertEqual(len(created_models), 3)
        config_by_model = {
            model.model_name: model.deal_configs["ltv_ratio"]
            for model in created_models
        }

        for cfg in config_by_model.values():
            self.assertEqual(cfg.mode, "continuous")
            self.assertEqual(cfg.transform, "log")

        self.assertTrue(config_by_model["M1_ltv_ratio_full"].include_main)
        self.assertTrue(config_by_model["M1_ltv_ratio_full"].include_interaction)
        self.assertFalse(config_by_model["M1_ltv_ratio_int_only"].include_main)
        self.assertTrue(config_by_model["M1_ltv_ratio_int_only"].include_interaction)
        self.assertTrue(config_by_model["M1_ltv_ratio_main_only"].include_main)
        self.assertFalse(config_by_model["M1_ltv_ratio_main_only"].include_interaction)


if __name__ == "__main__":
    unittest.main()
