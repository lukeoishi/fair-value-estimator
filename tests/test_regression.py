import unittest
import numpy as np
from regression import features,design,fit_predict

class RegressionTests(unittest.TestCase):
    def test_target_alignment(self):
        p=100*np.cumprod(1+np.arange(40)*.001)
        x,y,i=features(p)
        self.assertAlmostEqual(x[0,0],p[20]/p[19]-1)
        self.assertAlmostEqual(y[0],p[21]/p[20]-1)
        self.assertEqual(i[0],21)

    def test_future_does_not_change_features(self):
        p=np.arange(100.,150.)
        altered=p.copy();altered[30:]*=2
        np.testing.assert_allclose(features(p)[0][:10],features(altered)[0][:10])

    def test_recovers_quadratic(self):
        x=np.random.default_rng(0).normal(size=(200,3))
        y=1+2*x[:,0]-3*x[:,1]**2
        fit=np.arange(200)<150
        pred,_=fit_predict(x,y,fit,~fit,True)
        np.testing.assert_allclose(pred,y[~fit],atol=1e-10)
        self.assertEqual(design(x,True).shape[1],7)
