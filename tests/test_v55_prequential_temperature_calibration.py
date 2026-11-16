import unittest
import numpy as np
from scripts.v55_prequential_temperature_calibration import transform, select_temperature, mean_logloss

class TemperatureCalibrationTests(unittest.TestCase):
    def test_transform_preserves_six_expected_inclusions(self):
        p=np.linspace(0.05,0.25,45)
        for t in (0.5,1.0,1.5,2.0):
            adjusted=transform(p,t)
            self.assertEqual(adjusted.shape,(45,))
            self.assertTrue(np.all(adjusted>0))
            self.assertAlmostEqual(float(adjusted.sum()),6.0,places=8)

    def test_temperature_selection_is_deterministic_and_valid(self):
        rng=np.random.default_rng(42)
        p=np.vstack([np.full(45,6/45)+rng.normal(0,0.015,45) for _ in range(20)])
        p=np.clip(p,0.03,0.35)
        p=np.vstack([row*(6/row.sum()) for row in p])
        y=np.zeros((20,45))
        for i in range(20):
            y[i,rng.choice(45,6,replace=False)]=1
        a=select_temperature(p,y)
        b=select_temperature(p,y)
        self.assertEqual(a["temperature"],b["temperature"])
        self.assertGreater(a["temperature"],0)
        self.assertEqual(len(a["grid"]),61)

    def test_mean_logloss_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            mean_logloss(np.ones((2,44))*0.1,np.ones((2,45)))

if __name__=="__main__":
    unittest.main()
