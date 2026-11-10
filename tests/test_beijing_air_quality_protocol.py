import importlib
import unittest
import numpy as np


class TemporalProtocolTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('empirical_analysis.beijing_air_quality.protocol')
        except ModuleNotFoundError:self.fail('Beijing calendar protocol is missing')

    def test_calendar_outer_and_future_are_disjoint_and_available(self):
        api=self.api()
        dates=np.arange(np.datetime64('2013-03-02'),np.datetime64('2017-03-01')).astype(str)
        data=dict(dates=dates,input_dates=(dates.astype('datetime64[D]')-1).astype(str))
        splits=api.make_splits(data)
        self.assertEqual(len(splits),5)
        self.assertEqual(splits[0]['test_start'],'2014-03-01')
        self.assertEqual(splits[0]['test_end_exclusive'],'2014-09-01')
        self.assertEqual(splits[-1]['protocol'],'future')
        self.assertEqual(splits[-1]['test_start'],'2016-03-01')
        future_test=[]
        for split in splits:
            train=np.array(split['train']);test=np.array(split['test'])
            self.assertLess(max(dates[train]),min(dates[test]))
            self.assertEqual(len(split['inner']),3)
            for inner in split['inner']:
                a=np.array(inner['train']);b=np.array(inner['validation'])
                self.assertLess(max(dates[train[a]]),min(dates[train[b]]))
                self.assertLessEqual(max(dates[train[b]]),max(dates[train]))
            future_test.extend(test)
        self.assertEqual(len(future_test),len(set(future_test)))
        first=splits[0]['inner'][0]
        self.assertEqual(first['validation_start'],'2013-09-01')
        self.assertEqual(first['validation_end_exclusive'],'2013-11-01')

    def test_target_day_input_is_refused(self):
        api=self.api()
        dates=np.arange(np.datetime64('2013-03-02'),np.datetime64('2017-03-01')).astype(str)
        with self.assertRaisesRegex(ValueError,'previous'):
            api.make_splits(dict(dates=dates,input_dates=dates))


if __name__=='__main__':unittest.main()
