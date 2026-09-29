import unittest
import numpy as np
from harmonic import forecast, walk_forward


class HarmonicTests(unittest.TestCase):
    def test_recovers_phase_shifted_wave(self):
        t=np.arange(121)
        p=100+3*np.sin(2*np.pi*t/60+.7)
        self.assertAlmostEqual(forecast(p[:-1],(60,),30),p[-1],places=10)

    def test_future_cannot_change_past_predictions(self):
        p=100+np.arange(150)*.1
        changed=p.copy();changed[140:]+=30
        np.testing.assert_allclose(walk_forward(p,(60,),30)[:141],
                                   walk_forward(changed,(60,),30)[:141])

    def test_constant(self):
        self.assertAlmostEqual(forecast(np.full(120,100.)),100.,places=10)

    def test_invalid_half_life(self):
        with self.assertRaises(ValueError):
            forecast(np.full(120,100.),half_life=0)
