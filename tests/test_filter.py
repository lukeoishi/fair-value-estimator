import unittest
import numpy as np
from fair_value import kalman, ewma, simulate, MARKETS


class FilterTests(unittest.TestCase):
    def test_hand_calculated_update(self):
        estimate, variance, gain = kalman([100, 102], 1, 1)
        self.assertAlmostEqual(gain[1], 2 / 3)
        self.assertAlmostEqual(estimate[1], 101 + 1 / 3)
        self.assertAlmostEqual(variance[1], 2 / 3)

    def test_missing_quote_grows_uncertainty(self):
        estimate, variance, gain = kalman([100, np.nan, np.nan, 101], 1, 1)
        np.testing.assert_allclose(estimate[:3], 100)
        np.testing.assert_allclose(variance[:3], [1, 2, 3])
        self.assertAlmostEqual(gain[3], .8)

    def test_future_does_not_change_past(self):
        _, quotes = simulate(MARKETS[0])
        changed = quotes.copy()
        changed[200:] += 30
        np.testing.assert_array_equal(kalman(quotes, .02, 1)[0][:200],
                                      kalman(changed, .02, 1)[0][:200])

    def test_steady_state_matches_ewma(self):
        q, r = .0225, 1
        prior = (q + np.sqrt(q*q + 4*q*r)) / 2
        alpha = prior / (prior + r)
        _, quotes = simulate(MARKETS[0])
        np.testing.assert_allclose(kalman(quotes, q, r)[0][300:],
                                   ewma(quotes, alpha)[300:], atol=1e-10)

    def test_invalid_parameters(self):
        for q, r in [(-1, 1), (1, 0), (np.nan, 1)]:
            with self.assertRaises(ValueError):
                kalman([100, 101], q, r)


if __name__ == '__main__':
    unittest.main()
