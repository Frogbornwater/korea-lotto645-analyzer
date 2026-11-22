# -*- coding: utf-8 -*-
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from v93_multiscale_prequential import _forecast, analyze_v93, load_draws, _holm, _block_sign_flip_p

class V93Tests(unittest.TestCase):
    def test_forecast_sums_to_six_and_is_shrunk(self):
        hist=[[1,2,3,4,5,6] for _ in range(20)]
        p=_forecast(hist,20)
        self.assertAlmostEqual(float(p.sum()),6.0,places=12)
        self.assertGreater(p[0],6/45)
        self.assertLess(p[44],6/45)
    def test_strict_prequential_and_metrics_shape(self):
        draws=list(range(1,1002))
        nums=[[(i+j)%45+1 for j in range(6)] for i in range(1001)]
        result=analyze_v93(draws,nums,bootstrap_reps=1000)
        self.assertEqual(result["protocol"]["target_draws"],1)
        self.assertEqual(len(result["metrics"]),10)
        self.assertFalse(result["interpretation"]["predictive_advantage_established"])
    def test_rejects_short_or_duplicate_history(self):
        with self.assertRaisesRegex(ValueError,"至少需要1001期"):
            analyze_v93(list(range(1,1001)),[list(range(1,7)) for _ in range(1000)],1000)
        with self.assertRaisesRegex(ValueError,"唯一且严格升序"):
            analyze_v93([1,1]+list(range(3,1002)),[list(range(1,7)) for _ in range(1001)],1000)
    def test_block_sign_flip_is_deterministic(self):
        diffs=[-0.01,0.02,-0.03,0.01]*8
        self.assertEqual(_block_sign_flip_p(diffs,seed=7,reps=1000,block=4),
                         _block_sign_flip_p(diffs,seed=7,reps=1000,block=4))
    def test_holm_is_monotone_in_sorted_order(self):
        p={"a":.01,"b":.04,"c":.03}
        adj=_holm(p)
        self.assertGreaterEqual(adj["a"],.01)
        self.assertGreaterEqual(adj["b"],adj["c"])
if __name__=="__main__": unittest.main()
