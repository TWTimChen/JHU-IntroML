import unittest
import numpy as np
from src import evaluation_metrics as em

class TestEvaluationMetrics(unittest.TestCase):

    # Classification Metrics Tests
    def test_accuracy(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        result = em.accuracy(y_true, y_pred)
        self.assertEqual(result, 0.8)

    def test_precision(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        result = em.precision(y_true, y_pred)
        self.assertEqual(result, 1.0)

    def test_recall(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        result = em.recall(y_true, y_pred)
        self.assertEqual(result, 0.6666666666666666)

    def test_f1_score(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        result = em.f1_score(y_true, y_pred)
        self.assertEqual(result, 0.8)

    # Regression Metrics Tests
    def test_mean_squared_error(self):
        y_true = [3.5, 2.7, 4.6]
        y_pred = [3.6, 2.8, 4.5]
        result = em.mean_squared_error(y_true, y_pred)
        self.assertAlmostEqual(result, 0.01)

    def test_mean_absolute_error(self):
        y_true = [3.5, 2.7, 4.6]
        y_pred = [3.6, 2.8, 4.5]
        result = em.mean_absolute_error(y_true, y_pred)
        self.assertAlmostEqual(result, 0.1)

    def test_r2_coefficient(self):
        y_true = [3.5, 2.7, 4.6]
        y_pred = [3.6, 2.8, 4.5]
        result = em.r2_coefficient(y_true, y_pred)
        self.assertAlmostEqual(result, 0.98351648)

    def test_pearson_correlation(self):
        y_true = [3.5, 2.7, 4.6]
        y_pred = [3.6, 2.8, 4.5]
        result = em.pearson_correlation(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9983782)

if __name__ == '__main__':
    unittest.main()
