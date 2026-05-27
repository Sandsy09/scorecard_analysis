import unittest
import warnings

import pandas as pd

from preprocessing.binning import VariableBinner


class BinningTests(unittest.TestCase):
    def test_transform_allows_duplicate_woe_labels(self):
        df = pd.DataFrame({
            "x": [0] * 100 + [1] * 100,
            "target": [1] * 10 + [0] * 90 + [1] * 10 + [0] * 90,
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binner = VariableBinner("x", "target").fit_manual(df, [0.5])
            transformed = binner.transform(df)

        self.assertEqual(len(transformed), len(df))
        self.assertFalse(transformed.isna().any())
        self.assertEqual(transformed.nunique(), 1)


if __name__ == "__main__":
    unittest.main()

