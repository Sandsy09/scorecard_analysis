import unittest

import pandas as pd

from modelling.scorecard_scaler import ScorecardScaler


class _DummyModel:
    variables = ["x"]
    coefficients = pd.Series({"x": 1.0})
    intercept = 0.0


class ScorecardScalerTests(unittest.TestCase):
    def test_score_pd_round_trip_uses_base_odds(self):
        scaler = ScorecardScaler(pdo=20, base_score=600, base_odds=50)

        pd_at_base = scaler.score_to_pd(600)

        self.assertAlmostEqual(pd_at_base, 1 / 51, places=8)
        self.assertAlmostEqual(scaler.pd_to_score(pd_at_base), 600, places=8)

    def test_unmapped_woe_values_raise_instead_of_zero_points(self):
        scaler = ScorecardScaler()
        model = _DummyModel()
        bin_stats = {
            "x": pd.DataFrame([{
                "bin": "low",
                "woe": 0.1,
                "bad_rate": 0.2,
                "n_total": 100,
            }])
        }
        scaler.build(model, bin_stats)

        with self.assertRaisesRegex(ValueError, "Unable to map"):
            scaler.score(pd.DataFrame({"x_woe": [0.2]}), model, bin_stats)


if __name__ == "__main__":
    unittest.main()

