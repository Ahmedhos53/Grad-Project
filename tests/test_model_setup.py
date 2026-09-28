import unittest

from tools.setup_models import ALLOWED_HOSTS, _safe_destination, load_manifest


class ModelSetupTests(unittest.TestCase):
    def test_manifest_is_allowlisted_and_traversal_is_rejected(self):
        assets = load_manifest()

        self.assertEqual({asset["id"] for asset in assets}, {"yolov8n", "yolov8s", "yamnet", "primus_checkpoint"})
        self.assertTrue(all(str(asset["url"]).startswith("https://") for asset in assets))
        self.assertTrue(all(asset["url"].split("/", 3)[2] in ALLOWED_HOSTS for asset in assets))
        with self.assertRaises(ValueError):
            _safe_destination("../outside-model.bin")


if __name__ == "__main__":
    unittest.main()
