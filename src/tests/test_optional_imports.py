import subprocess
import sys
import textwrap
import unittest


class OptionalImportTests(unittest.TestCase):
    def test_tensor_abess_imports_without_sklearn(self) -> None:
        code = textwrap.dedent(
            """
            import tensor_abess
            assert hasattr(tensor_abess, "tensor_abess")
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"import failed with stderr:\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
