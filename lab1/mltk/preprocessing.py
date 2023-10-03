import pandas as pd
import numpy as np
from typing import Dict, List, Union

class DataPipeline:
    def __init__(self, steps: List[Dict[str, Union[str, Dict]]]):
        """
        Initialize the data pipeline.

        Parameters:
        - steps: A list of dictionary specifying the steps to be executed. Each step is a dictionary with:
          - 'operation': The name of the operation.
          - 'params': A dictionary with parameters for the operation.
        
        Usage Example:

        steps = [
            {"operation": "fill_missing_values"},
            {"operation": "handle_ordinal_data","params": {"column": "ordinal_column_name", "mapping": {'Low': 1, 'Medium': 2, 'High': 3}}},
            {"operation": "one_hot_encode", "params": {"column": "nominal_column_name"}},
            {"operation": "standardize", "params": {"column": "numeric_column_name"}}
        ]

        pipeline = DataPipeline(steps)
        df = pipeline.run(df)
        """
        self.steps = steps

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace non-standard missing values with NaN
        na_values = {"?", "x", " ", "", "NaN"}
        for value in na_values:
            df.replace(value, np.nan, inplace=True)
        
        # Try converting 'object' type columns to numeric
        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    df[column] = pd.to_numeric(df[column])
                except ValueError:
                    pass  # If conversion fails, let the column remain as 'object' type
        
        # Fill missing values with the mean of each column
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:  # only fill numeric columns
                df[column].fillna(df[column].mean(), inplace=True)
            elif df[column].dtype == 'object':
                df[column].fillna(df[column].mode(), inplace=True)
        
        return df

    def _handle_ordinal_data(self, df: pd.DataFrame, column: str, mapping: Dict[str, Union[int, float]]) -> pd.DataFrame:
        df[column] = df[column].replace(mapping).astype('int64')
        return df

    def _one_hot_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        return pd.get_dummies(df, columns=[column], drop_first=True)

    def _discretize_equal_width(self, df: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
        df[column] = pd.cut(df[column], bins=bins)
        return df

    def _discretize_equal_frequency(self, df: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
        df[column] = pd.qcut(df[column], q=bins)
        return df

    def _standardize(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df[column] = (df[column] - np.mean(df[column])) / np.std(df[column], ddof=1)
        return df
    
    def _function_transformer(self, df: pd.DataFrame, column: str, func: callable) -> pd.DataFrame:
        df[column] = df[column].apply(func)
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the pipeline on the provided DataFrame.

        Parameters:
        - df: The DataFrame to process.

        Returns:
        - Processed DataFrame.
        """
        for step in self.steps:
            operation = step['operation']
            params = step.get('params', {})
            
            # Using a dictionary mapping for better readability and scalability
            operations = {
                "fill_missing_values": self._fill_missing_values,
                "handle_ordinal_data": self._handle_ordinal_data,
                "one_hot_encode": self._one_hot_encode,
                "discretize_equal_width": self._discretize_equal_width,
                "discretize_equal_frequency": self._discretize_equal_frequency,
                "standardize": self._standardize,
                "function_transformer": self._function_transformer
            }

            # Get the function to execute from the dictionary and call it
            if operation in operations:
                function_to_execute = operations[operation]
                df = function_to_execute(df, **params)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        return df
