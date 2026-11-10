import importlib.util
from pathlib import Path
import tempfile
import unittest


class BeijingControllerTests(unittest.TestCase):
    def api(self):
        path=Path('doc/beijing_air_quality/local_execute.py')
        self.assertTrue(path.exists(),'Beijing local controller is missing')
        spec=importlib.util.spec_from_file_location('beijing_controller',path)
        result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
        return result

    def test_command_and_immutable_progress(self):
        api=self.api()
        command=api.worker_command('python.exe','path with space/config.json',34)
        self.assertIn('empirical_analysis.beijing_air_quality.run',command)
        self.assertIn('path with space/config.json',command)
        with self.assertRaises(ValueError):api.worker_command('p','c',35)
        with tempfile.TemporaryDirectory() as folder:
            first=api.publish_status(folder,dict(completed=1));original=first.read_bytes()
            with first.open('rb') as reader:
                second=api.publish_status(folder,dict(completed=2))
                self.assertNotEqual(first,second)
                self.assertEqual(first.read_bytes(),original)
                self.assertEqual(reader.read(),original)


if __name__=='__main__':unittest.main()
