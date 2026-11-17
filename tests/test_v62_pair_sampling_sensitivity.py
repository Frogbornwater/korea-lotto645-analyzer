import unittest

from scripts.v62_pair_sampling_sensitivity import (
    DATA_LAST_DRAW, FOLD_SIZE, FOLD_STARTS, PAIR_STRENGTHS, SEEDS, validate_design,
)


class V62PairSamplingSensitivityTests(unittest.TestCase):
    def test_predeclared_strength_grid_and_seeds(self):
        validate_design()
        self.assertEqual(PAIR_STRENGTHS, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(len(set(SEEDS)), 5)

    def test_target_folds_are_disjoint_and_inside_development_range(self):
        folds = [(s, s + FOLD_SIZE - 1) for s in FOLD_STARTS]
        self.assertEqual(folds, [(850, 899), (900, 949), (950, 999), (1000, 1049)])
        self.assertTrue(all(end <= DATA_LAST_DRAW for _, end in folds))
        self.assertTrue(all(b[0] > a[1] for a, b in zip(folds, folds[1:])))


if __name__ == "__main__":
    unittest.main()
