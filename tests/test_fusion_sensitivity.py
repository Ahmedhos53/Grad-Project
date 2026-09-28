import unittest

from tools.evaluate_fusion_sensitivity import COMPONENTS, evaluate


class FusionSensitivityTests(unittest.TestCase):
    def test_synthetic_baseline_and_ablation_comparisons_are_explicitly_bounded(self):
        report = evaluate()
        baseline = report["baseline_comparison"]
        ablation = report["ablation_sensitivity"]

        self.assertEqual(baseline["scenario_count"], len(report["combination_sensitivity"]))
        self.assertEqual(baseline["scenario_count"], 130)
        self.assertEqual(baseline["independent_or_any_signal"]["trigger_count"], 129)
        self.assertLessEqual(
            baseline["current_any_nonzero_score"]["trigger_count"],
            baseline["independent_or_any_signal"]["trigger_count"],
        )
        self.assertEqual(ablation["comparison_count"], 324)
        self.assertEqual(set(ablation["per_component"]), set(COMPONENTS))
        self.assertGreater(
            ablation["per_component"]["speech_phone"]["mean_score_reduction"],
            0,
        )
        self.assertIn("not alert accuracy", baseline["interpretation"])
        self.assertIn("not a labelled behavioural ablation", ablation["interpretation"])

    def test_dependency_aware_scenarios_match_phone_and_speech_rules(self):
        report = evaluate()
        dependencies = report["signal_dependencies"]
        valid = dependencies["dependency_valid_combinations"]

        self.assertLess(valid["scenario_count"], 130)
        self.assertGreater(valid["scenario_count"], 0)
        for row in valid["combinations"]:
            active = set(row["active_components"])
            self.assertFalse("phone_call" in active and "phone_object" not in active)
            self.assertFalse("speech_phone" in active and "phone_call" not in active)
        phone_cases = {row["case"]: row for row in report["context_sensitivity"]}
        self.assertEqual(phone_cases["phone_without_speech"]["current_score"], 48)
        self.assertEqual(phone_cases["phone_with_speech"]["current_score"], 54)
        self.assertIn("prerequisite", valid["ablation"]["removal_rule"])


if __name__ == "__main__":
    unittest.main()
