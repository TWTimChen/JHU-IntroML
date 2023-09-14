import unittest
import numpy as np
import pandas as pd
from src.cross_validation import train_test_split, k_fold_split

class TestTrainTestSplit(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe
        self.df = pd.DataFrame({
            'feature1': np.arange(100),
            'feature2': np.random.rand(100),
            'target': np.random.choice([0, 1], size=100)
        })

    def test_size(self):
        train, test = train_test_split(self.df, test_size=0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_stratified_split(self):
        train, test = train_test_split(self.df, test_size=0.2, stratified=True)
        original_dist = self.df['target'].value_counts(normalize=True)
        train_dist = train['target'].value_counts(normalize=True)
        test_dist = test['target'].value_counts(normalize=True)

        # Assert that the distribution difference is small
        for label in original_dist.index:
            self.assertAlmostEqual(original_dist[label], train_dist[label], delta=0.1)
            self.assertAlmostEqual(original_dist[label], test_dist[label], delta=0.1)

    def test_no_overlap(self):
        train, test = train_test_split(self.df, test_size=0.2)
        # There should be no common indices between train and test sets
        common_indices = train.index.intersection(test.index)
        self.assertEqual(len(common_indices), 0)

class TestKFoldSplit(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe
        self.df = pd.DataFrame({
            'feature1': np.arange(100),
            'feature2': np.random.rand(100),
            'target': np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        })

    def test_num_splits(self):
        splits = list(k_fold_split(self.df, n_splits=5))
        self.assertEqual(len(splits), 5)

    def test_stratified_split(self):
        splits = list(k_fold_split(self.df, n_splits=5, stratified=True))
        original_dist = self.df['target'].value_counts(normalize=True)

        for train, test in splits:
            train_dist = train['target'].value_counts(normalize=True)
            test_dist = test['target'].value_counts(normalize=True)

            # Assert that the distribution difference is small
            for label in original_dist.index:
                self.assertAlmostEqual(original_dist[label], train_dist[label], delta=0.1)
                self.assertAlmostEqual(original_dist[label], test_dist[label], delta=0.1)

    def test_no_overlap(self):
        splits = list(k_fold_split(self.df, n_splits=5))
        common_indices = []

        # Check that there's no overlap between all train indices and all test indices
        for train, test in splits:
            common_indices.extend(set(train.index).intersection(test.index))

        self.assertEqual(len(common_indices), 0)

    def test_split_size_non_stratified(self):
        splits = list(k_fold_split(self.df, n_splits=5, stratified=False))
        for train, test in splits:
            self.assertTrue(len(test) == len(self.df) / 5)
            self.assertTrue(len(train) == len(self.df) * 4 / 5)

if __name__ == '__main__':
    unittest.main()
