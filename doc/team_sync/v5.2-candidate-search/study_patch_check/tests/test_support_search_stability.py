"""Independent training-only multistart support-search contracts."""
from __future__ import annotations
import importlib.util
import inspect
import unittest
import numpy as np

class SupportStabilityTests(unittest.TestCase):
    def module(self):
        path=__import__('pathlib').Path(__file__).resolve().parents[1]/'numerical_simulation/candidate_search_study/stability.py'
        self.assertTrue(path.is_file(),'Independent support-stability implementation is missing')
        spec=importlib.util.spec_from_file_location('stability',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def sample(self):
        rng=np.random.default_rng(809)
        X=rng.normal(size=(90,4,4,3));B=np.zeros((4,4,3));B[np.ix_([0,2],[1],[0,2])]=1
        y=np.einsum('nijk,ijk->n',X,B)+rng.normal(scale=.05,size=90)
        return X,y

    def test_no_true_or_test_inputs_to_search(self):
        f=self.module().fit_multistart
        args=set(inspect.signature(f).parameters)
        self.assertFalse(args.intersection({'B_true','supports_true','X_test','y_test'}))

    def test_predeclared_starts_are_reproducible_and_fixed_size(self):
        m=self.module();X,y=self.sample()
        a=m.make_initializations(X,y,(2,1,2),training_seed=71)
        b=m.make_initializations(X,y,(2,1,2),training_seed=71)
        self.assertEqual([x['strategy'] for x in a],['marginal','random_0','random_1'])
        for row,other in zip(a,b):
            for s,t,size,dim in zip(row['supports'],other['supports'],(2,1,2),X.shape[1:]):
                np.testing.assert_array_equal(s,t);self.assertEqual(len(s),size)
                self.assertEqual(len(np.unique(s)),size)
                self.assertTrue(np.all((s>=0)&(s<dim)))

    def test_best_training_rss_is_selected_and_ols_consistent(self):
        m=self.module();X,y=self.sample()
        selected,records=m.fit_multistart(X,y,(2,1,2),training_seed=71,max_cycles=4,k_max=1,m_max=15)
        self.assertEqual(len(records),3)
        self.assertAlmostEqual(selected['rss'],min(r['rss'] for r in records),places=10)
        residual=y-np.einsum('nijk,ijk->n',X,selected['B'])
        self.assertAlmostEqual(selected['rss'],float(residual@residual),places=10)
        self.assertLessEqual(selected['rss'],records[0]['rss']+1e-12)
        for r in records:
            self.assertTrue(np.all(np.diff(r['loss_history'])<=1e-12))
            self.assertAlmostEqual(r['loss'],min(r['visited_loss_history']),places=12)

    def test_search_does_not_mutate_training_data(self):
        m=self.module();X,y=self.sample();a=X.copy();b=y.copy()
        m.fit_multistart(X,y,(2,1,2),training_seed=71,max_cycles=2,k_max=1,m_max=15)
        np.testing.assert_array_equal(X,a);np.testing.assert_array_equal(y,b)

    def test_full_search_is_reproducible(self):
        m=self.module();X,y=self.sample()
        a,ar=m.fit_multistart(X,y,(2,1,2),training_seed=71,max_cycles=3,k_max=1,m_max=15)
        b,br=m.fit_multistart(X,y,(2,1,2),training_seed=71,max_cycles=3,k_max=1,m_max=15)
        np.testing.assert_array_equal(a['B'],b['B'])
        self.assertEqual([r['rss'] for r in ar],[r['rss'] for r in br])

if __name__=='__main__':unittest.main()
