import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_public_entry_points_import(self):
        from interaction_pipeline import InteractionScorecardPipeline
        from modelling.deal_variable_plots import DealVariablePlotter
        from modelling.interaction_model import DealVariableConfig
        from pipeline import DataSplitConfig, ScorecardPipeline
        from stakeholder_evidence import StakeholderEvidenceReport
        from validation.metrics import ValidationReport

        self.assertIsNotNone(ScorecardPipeline)
        self.assertIsNotNone(DataSplitConfig)
        self.assertIsNotNone(InteractionScorecardPipeline)
        self.assertIsNotNone(DealVariableConfig)
        self.assertIsNotNone(DealVariablePlotter)
        self.assertIsNotNone(StakeholderEvidenceReport)
        self.assertIsNotNone(ValidationReport)


if __name__ == "__main__":
    unittest.main()

