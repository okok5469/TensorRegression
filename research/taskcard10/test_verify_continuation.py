"""Pure continuation-verifier gates: no simulation input or fitting."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class ContinuationAuditTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec('research.taskcard10.verify_continuation')
        self.assertIsNotNone(spec, 'Independent continuation verifier is not implemented')
        return __import__('research.taskcard10.verify_continuation', fromlist=['verify_continuation'])

    def protocol(self):
        old = {'formal': {'workers':4,'wall_time_limit_seconds':7200,'case_timeout_seconds':900,'replicates':2},
               'source_files':{'solver':'same'},'source_identity':{'solver':'same'},'threshold':.95}
        new = copy.deepcopy(old)
        new['formal'].update(workers=2,wall_time_limit_seconds=28800,case_timeout_seconds=3600)
        new['continuation'] = {'global_clock_reset_allowed':False}
        return old,new

    def test_compute_only_amendment_accepts_bounded_eight_hours(self):
        old,new = self.protocol()
        self.module().assert_protocol_amendment(old,new)

    def test_science_or_source_change_is_rejected(self):
        mod = self.module()
        for field in ('threshold','source_identity'):
            old,new = self.protocol();new[field] = .99 if field=='threshold' else {'solver':'changed'}
            with self.assertRaises(AssertionError):mod.assert_protocol_amendment(old,new)

    def test_shared_clock_does_not_accept_reset_or_late_completion(self):
        mod = self.module()
        clock = {'started_epoch':1000.,'limit_seconds':28800}
        mod.assert_clock_bounds(clock,[1001.,29000.])
        with self.assertRaises(AssertionError):mod.assert_clock_bounds(clock,[30000.])
        with self.assertRaises(AssertionError):mod.assert_clock_bounds(dict(clock,limit_seconds=7200),[1001.])

    def test_byte_exact_copy_rejects_changed_file(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as temp:
            first,second = Path(temp)/'a',Path(temp)/'b'
            first.mkdir();second.mkdir()
            (first/'evidence.json').write_text('original')
            (second/'evidence.json').write_text('original')
            identity = mod.file_manifest(first)
            mod.assert_copy_identity(first,second,identity)
            (second/'evidence.json').write_text('changed')
            with self.assertRaises(AssertionError):mod.assert_copy_identity(first,second,identity)

    def state_checker(self):
        mod = self.module()
        self.assertTrue(hasattr(mod,'confirmation_execution_state'), 'Zero-completion confirmation state is not implemented')
        return mod.confirmation_execution_state

    def test_zero_success_failed_confirmation_is_started_and_stopped(self):
        check = self.state_checker()
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            (raw/'stopped_confirmation.json').write_text(json.dumps({'status':'FAILED','completed':[],
                'failed':[{'task_id':0,'reason':'launch_failed'}],'not_started':list(range(1,48))}))
            self.assertEqual(check(raw,0),'STOPPED_PARTIAL')

    def test_never_started_confirmation_is_not_run(self):
        check = self.state_checker()
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            (raw/'confirmation_NOT_RUN.json').write_text(json.dumps({'status':'NOT_RUN','reason':'entry gate failed'}))
            self.assertEqual(check(raw,0),'NOT_RUN')

    def test_conflicting_stopped_and_not_run_markers_are_rejected(self):
        check = self.state_checker()
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            (raw/'stopped_confirmation.json').write_text(json.dumps({'status':'FAILED'}))
            (raw/'confirmation_NOT_RUN.json').write_text(json.dumps({'status':'NOT_RUN','reason':'wrong'}))
            with self.assertRaises(AssertionError):check(raw,0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
