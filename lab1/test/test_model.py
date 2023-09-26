import unittest
import numpy as np
from mltk.models import NullModel

class TestNullModel(unittest.TestCase):

    def test_classification(self):
        # Create data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        # Create and fit null model
        model = NullModel(task='classification')
        model.fit(X, y)

        # Since there are equal number of classes 0 and 1, it might pick any class
        # We just want to make sure it picks one of them
        self.assertIn(model.predict(X)[0], [0, 1])

    def test_classification_majority_class(self):
        # Create data where class 0 is the majority
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 0, 1])

        # Create and fit null model
        model = NullModel(task='classification')
        model.fit(X, y)

        # It should always predict the majority class, which is 0
        np.testing.assert_array_equal(model.predict(X), np.array([0, 0, 0, 0]))

    def test_regression(self):
        # Create data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1.5, 2.5, 3.5, 4.5])

        # Create and fit null model
        model = NullModel(task='regression')
        model.fit(X, y)

        # It should always predict the mean of y, which is 3.0
        np.testing.assert_array_equal(model.predict(X), np.full(X.shape[0], 3.0))

    def test_invalid_task(self):
        # Invalid task should default to regression
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1.5, 2.5, 3.5, 4.5])

        model = NullModel(task='invalid_task_name')
        model.fit(X, y)

        # It should always predict the mean of y, which is 3.0
        np.testing.assert_array_equal(model.predict(X), np.full(X.shape[0], 3.0))

if __name__ == '__main__':
    unittest.main()
