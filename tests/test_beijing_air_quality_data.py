import importlib.util
import unittest
from pathlib import Path
import numpy as np


def module():
    name='empirical_analysis.beijing_air_quality.data'
    try: spec=importlib.util.find_spec(name)
    except ModuleNotFoundError: spec=None
    assert spec is not None,'Beijing data module missing'
    return __import__(name,fromlist=['data'])


class BeijingDataTests(unittest.TestCase):
    def test_bins_preserve_station_pollutant_hour_axes_and_min_two_rule(self):
        d=module();hours=np.arange(2*48*6,dtype=float).reshape(2,48,6)
        hours[0,:3,0]=np.nan
        values,counts=d.aggregate_four_hours(hours)
        self.assertEqual(values.shape,(2,2,6,6))
        self.assertTrue(np.isnan(values[0,0,0,0]))
        self.assertEqual(counts[0,0,0,0],1)
        self.assertEqual(values[1,1,2,3],hours[1,24+12:24+16,2].mean())

    def test_target_and_persistence_use_observed_means_and_common_dates(self):
        d=module();hours=np.ones((4,24))*10
        hours[1,:6]=np.nan;hours[1,6:]=20
        hours[2,:7]=np.nan;hours[2,7:]=30
        result=d.daily_targets(hours.reshape(-1))
        self.assertEqual(result['target_counts'].tolist(),[18,17,24])
        self.assertEqual(result['previous_counts'].tolist(),[24,18,17])
        self.assertEqual(result['eligible'].tolist(),[True,False,False])
        self.assertEqual(result['y'][0],20.)
        self.assertEqual(result['persistence'][0],10.)


if __name__=='__main__':unittest.main()
