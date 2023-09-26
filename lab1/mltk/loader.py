import os
import pandas as pd
from typing import Optional

def load_data(filepath: str, verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file.

    Parameters:
    - filepath: Path to the CSV file.
    - verbose: Whether to print out the error message if the file does not exist.

    Returns:
    - DataFrame containing the loaded data, or None if file does not exist.
    """
    
    # Check if file exists
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        if verbose:
            print(f"Error: The file {filepath} does not exist.")
        return None
