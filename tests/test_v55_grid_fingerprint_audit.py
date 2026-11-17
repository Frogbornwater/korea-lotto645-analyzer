import unittest
import numpy as np
import pandas as pd
from scripts.v55_build_audit_bundle import canonical_history_sha256, build_forward_lock
from scripts.v55_prequential_temperature_calibration import select_temperature, TEMPERATURE_GRID

class V55GridAndFingerprintTests(unittest.TestCase):
    def test_temperature_grid_expanded_and_boundary_is_reported(self):
        self.assertAlmostEqual(TEMPERATURE_GRID[0], 0.25)
        self.assertAlmostEqual(TEMPERATURE_GRID[-1], 4.0)
        rng=np.random.default_rng(77)
        p=np.tile(np.full(45,6/45,dtype=float),(12,1))
        y=np.zeros((12,45),dtype=float)
        for i in range(12):
            y[i,rng.choice(45,6,replace=False)]=1
        result=select_temperature(p,y)
        self.assertIn("grid_boundary_optimum",result)
        self.assertIn(result["grid_status"],("interior_grid_optimum","boundary_optimum_requires_wider_search"))

    def test_fingerprint_uses_canonical_validated_rows(self):
        rows=[]
        for draw in range(1,4):
            nums=[draw,draw+5,draw+10,draw+15,draw+20,draw+25]
            rows.append({"draw_no":draw,"date":"20260101","n1":nums[0],"n2":nums[1],"n3":nums[2],
                         "n4":nums[3],"n5":nums[4],"n6":nums[5],"bonus":45})
        canonical=pd.DataFrame(rows)
        shuffled=canonical.iloc[::-1].copy()
        shuffled["date"]="2026-01-01"
        self.assertEqual(canonical_history_sha256(canonical),canonical_history_sha256(shuffled))

if __name__=="__main__":
    unittest.main()
