from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/v4"))
from layered_47_interactive_proxy_checkpoint import write_rejected_diagnostic_atomic


class RejectedDiagnosticTest(unittest.TestCase):
    def test_preserves_bad_values_and_explicitly_denies_resume(self):
        arrays = {"history": np.array([[np.nan, np.inf], [-np.inf, 7.]]),
                  "ids": np.array(["r1-rock-000"])}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rejected-raw.npz"
            result = write_rejected_diagnostic_atomic(p, arrays, "nonfinite")
            with np.load(p, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["history"], arrays["history"])
                np.testing.assert_array_equal(saved["ids"], arrays["ids"])
            self.assertEqual(result["nonfinite_fields"]["history"]["count"], 3)
            self.assertFalse(result["resume_qualified"])
            self.assertFalse(result["training_eligible"])
            self.assertEqual(json.loads(Path(str(p)+".json").read_text()), result)
            with self.assertRaises(FileExistsError):
                write_rejected_diagnostic_atomic(p, arrays, "nonfinite")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_rejects_object_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rejected-raw.npz"
            with self.assertRaisesRegex(ValueError, "object_dtype"):
                write_rejected_diagnostic_atomic(p, {"bad": np.array([{}])}, "shape")
            self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()
