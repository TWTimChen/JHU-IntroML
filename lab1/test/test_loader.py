import unittest
import pandas as pd
import tempfile
import os
from mltk.loader import load_data

class TestLoadData(unittest.TestCase):

    def test_load_data_valid_path(self):
        # Create a temporary CSV file
        data = """name,age
        Alice,28
        Bob,22
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(data.encode('utf-8'))
            
        # Use load_data to read the temporary CSV file
        df = load_data(tmp.name)
        
        # Check if DataFrame was loaded correctly
        self.assertEqual(len(df), 2)
        self.assertTrue('name' in df.columns)
        self.assertTrue('age' in df.columns)

        # Cleanup: Remove the temporary CSV file
        os.remove(tmp.name)

    def test_load_data_invalid_path(self):
        # An invalid path
        invalid_path = "/path/that/does/not/exist.csv"
        
        # Expecting the function to return None
        self.assertIsNone(load_data(invalid_path, verbose=False))

if __name__ == '__main__':
    unittest.main()
