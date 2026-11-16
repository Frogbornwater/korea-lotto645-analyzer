import unittest
import numpy as np
import pandas as pd
from scripts.v55_build_audit_bundle import build_forward_lock

class V55ForwardLockTests(unittest.TestCase):
    @staticmethod
    def history(n=280, seed=55):
        rng=np.random.default_rng(seed)
        rows=[]
        for draw in range(1,n+1):
            nums=sorted(rng.choice(np.arange(1,46),6,replace=False).tolist())
            rows.append({"draw_no":draw,"date":"","n1":nums[0],"n2":nums[1],"n3":nums[2],
                         "n4":nums[3],"n5":nums[4],"n6":nums[5],"bonus":int(next(x for x in range(1,46) if x not in nums))})
        return pd.DataFrame(rows)

    def test_forward_lock_is_frozen_after_last_calibration_draw(self):
        lock=build_forward_lock(self.history())
        self.assertTrue(lock["frozen"])
        self.assertEqual(lock["base_last_draw"],280)
        self.assertEqual(lock["calibration_first_draw"],231)
        self.assertEqual(lock["calibration_last_draw"],280)
        self.assertGreater(lock["temperature"],0)
        self.assertEqual(len(lock["history_sha256"]),64)
        self.assertIn("strictly greater",lock["activation_rule"])

    def test_forward_lock_rejects_insufficient_history(self):
        with self.assertRaises(ValueError):
            build_forward_lock(self.history(220))

if __name__=="__main__":
    unittest.main()
