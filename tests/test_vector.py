import unittest
import numpy as np

from dedupe.variables.vector import (
    VectorType,
    cosine_distance,
    euclidean_distance,
    vector_sign_hash_predicate,
)


class TestVectorVariable(unittest.TestCase):
    def test_cosine_distance_identical(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        # Identical vectors should have distance 0.0
        dist = cosine_distance(v1, v2)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_cosine_distance_orthogonal(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        # Orthogonal vectors have cos_sim = 0.0, normalized distance = 0.5
        dist = cosine_distance(v1, v2)
        self.assertAlmostEqual(dist, 0.5, places=5)

    def test_cosine_distance_opposite(self):
        v1 = [1.0, 2.0]
        v2 = [-1.0, -2.0]
        # Opposite vectors have cos_sim = -1.0, normalized distance = 1.0
        dist = cosine_distance(v1, v2)
        self.assertAlmostEqual(dist, 1.0, places=5)

    def test_cosine_distance_missing_or_empty(self):
        self.assertTrue(np.isnan(cosine_distance(None, [1.0, 2.0])))
        self.assertTrue(np.isnan(cosine_distance([1.0, 2.0], None)))
        self.assertTrue(np.isnan(cosine_distance([], [])))
        self.assertTrue(np.isnan(cosine_distance([1.0], [1.0, 2.0])))

    def test_euclidean_distance(self):
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        dist = euclidean_distance(v1, v2)
        self.assertAlmostEqual(dist, 5.0, places=5)

    def test_vector_sign_hash_predicate(self):
        v = [1.5, -2.0, 0.8, -0.1, 4.2, -3.1, 0.5, -0.9, 1.1, -1.2, 0.3, -0.4, 2.0, -1.0, 0.7, -0.8]
        preds = vector_sign_hash_predicate(v)
        self.assertIsInstance(preds, tuple)
        self.assertGreater(len(preds), 0)
        # Should be deterministic
        self.assertEqual(preds, vector_sign_hash_predicate(v))

    def test_vector_type_initialization(self):
        vec_var = VectorType("embedding", metric="cosine", has_missing=True)
        self.assertEqual(vec_var.field, "embedding")
        self.assertEqual(vec_var.type, "Vector")
        self.assertTrue(vec_var.has_missing)
        self.assertEqual(vec_var.comparator, cosine_distance)
        self.assertIn(vector_sign_hash_predicate, vec_var._predicate_functions)

    def test_vector_type_euclidean(self):
        vec_var = VectorType("feature_vec", metric="euclidean")
        self.assertEqual(vec_var.comparator, euclidean_distance)

    def test_vector_type_invalid_metric(self):
        with self.assertRaises(ValueError):
            VectorType("embedding", metric="manhattan")


if __name__ == "__main__":
    unittest.main()
