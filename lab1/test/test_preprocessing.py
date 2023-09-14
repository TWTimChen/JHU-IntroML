import unittest
import pandas as pd
from src.preprocessing import DataPipeline

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, '?', 5],
            'B': ['Low', 'Medium', 'High', 'Low', 'Medium'],
            'C': ['Red', 'Green', 'Red', 'Green', 'Red'],
            'D': [0.5, 0.7, 0.2, 0.4, 0.9],
        })

    def test_fill_missing_values(self):
        pipeline = DataPipeline([{"operation": "fill_missing_values"}])
        df = pipeline.run(self.data)
        self.assertEqual(df['A'].isnull().sum(), 0)

    def test_handle_ordinal_data(self):
        pipeline = DataPipeline([{"operation": "handle_ordinal_data", "params": {"column": "B", "mapping": {'Low': 1, 'Medium': 2, 'High': 3}}}])
        df = pipeline.run(self.data)
        self.assertTrue((df['B'] == [1, 2, 3, 1, 2]).all())

    def test_one_hot_encode(self):
        pipeline = DataPipeline([{"operation": "one_hot_encode", "params": {"column": "C"}}])
        df = pipeline.run(self.data)
        self.assertTrue(('C_Red' in df.columns) ^ ('C_Green' in df.columns))

    def test_discretize_equal_width(self):
        pipeline = DataPipeline([{"operation": "discretize_equal_width", "params": {"column": "D", "bins": 2}}])
        df = pipeline.run(self.data)
        self.assertTrue(isinstance(df['D'].iloc[0], pd.Interval))

    def test_discretize_equal_frequency(self):
        pipeline = DataPipeline([{"operation": "discretize_equal_frequency", "params": {"column": "D", "bins": 2}}])
        df = pipeline.run(self.data)
        self.assertTrue(isinstance(df['D'].iloc[0], pd.Interval))

    def test_standardize(self):
        pipeline = DataPipeline([{"operation": "standardize", "params": {"column": "D"}}])
        df = pipeline.run(self.data)
        self.assertAlmostEqual(df['D'].mean(), 0, places=5)
        self.assertAlmostEqual(df['D'].std(), 1, places=5)

    def test_unknown_operation(self):
        pipeline = DataPipeline([{"operation": "unknown_operation"}])
        with self.assertRaises(ValueError):
            df = pipeline.run(self.data)

if __name__ == '__main__':
    unittest.main()
