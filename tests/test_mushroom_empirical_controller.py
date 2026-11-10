import importlib.util
from pathlib import Path
import unittest
import ctypes
import json
import os
import tempfile


class ControllerTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Requires real Windows sharing semantics')
    def test_status_publication_does_not_replace_reader_held_file(self):
        spec=importlib.util.spec_from_file_location('mushroom_controller','doc/mushroom_empirical/local_execute.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        create=kernel.CreateFileW
        create.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
        create.restype=ctypes.c_void_p
        close=kernel.CloseHandle;close.argtypes=[ctypes.c_void_p];close.restype=ctypes.c_int
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            target=directory/'STATUS.json';target.write_text('{"status":"OLD"}')
            handle=create(str(target),0x80000000,1,None,3,0x80,None)
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            try:
                replacement=directory/'STATUS.tmp';replacement.write_text('{}')
                with self.assertRaises(PermissionError):
                    replacement.replace(target)
                self.assertTrue(callable(getattr(module,'publish_status',None)),
                                'Status publisher must tolerate Windows readers by creating immutable snapshots')
                first=module.publish_status(directory,dict(status='RUNNING',completed=40))
                second=module.publish_status(directory,dict(status='FAILED',error='real worker failure'))
                self.assertNotEqual(first,second)
                self.assertEqual(json.loads(first.read_text())['completed'],40)
                self.assertEqual(json.loads(second.read_text())['error'],'real worker failure')
                self.assertEqual(json.loads(target.read_text())['status'],'OLD')
                self.assertEqual(len(list((directory/'status_events').glob('*.json'))),2)
            finally:
                close(handle)

    def test_command_uses_mushroom_and_preserves_config_as_one_argument(self):
        path=Path('doc/mushroom_empirical/local_execute.py')
        self.assertTrue(path.exists(),'Task15 local controller is missing')
        spec=importlib.util.spec_from_file_location('mushroom_controller',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.worker_command('python.exe','some path/config.json',159)
        self.assertIn('empirical_analysis.spectrofood_mushroom.run',result)
        self.assertIn('some path/config.json',result)
        self.assertEqual(result[-1],'159')
        with self.assertRaises(ValueError):module.worker_command('p','c',160)


if __name__=='__main__':unittest.main()
