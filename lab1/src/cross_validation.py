import numpy as np
import pandas as pd
from typing import Tuple, Iterator

def train_test_split(df: pd.DataFrame, test_size: float=0.2, stratified: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train and test sets.

    Parameters:
    - df: DataFrame to be split.
    - test_size: Proportion of the dataset to include in the test split.
    - stratified: Whether to perform stratified sampling. If True, the function will ensure that the train 
                  and test set have roughly the same proportion of samples of each target class as the complete set. 
                  Assumes the target column is named 'target'.

    Returns:
    - Train and test DataFrames.
    """
    
    # Array of indices of the dataframe
    indices = np.arange(len(df))
    
    if stratified:
        # Get target values from the DataFrame
        labels = df['target'].values
        
        # Get unique labels and their counts in the dataset
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        # Calculate the proportion of each label in the dataset
        fractions = label_counts / len(labels)
        
        test_indices = np.array([], dtype=int)
        # For each unique label, determine the number of instances to include in the test set
        # based on the fraction of that label in the original dataset
        for label, fraction in zip(unique_labels, fractions):
            label_indices = indices[labels == label]
            label_test_indices = np.random.choice(label_indices, size=int(test_size * len(df) * fraction), replace=False)
            test_indices = np.concatenate([test_indices, label_test_indices])
        
        # The train set consists of all indices not in the test set
        train_indices = np.setdiff1d(indices, test_indices)
        return df.iloc[train_indices], df.iloc[test_indices]
    
    else:
        # For non-stratified splits, just shuffle the data randomly
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_size))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]
        return df.iloc[train_indices], df.iloc[test_indices]

def k_fold_split(df: pd.DataFrame, n_splits: int=5, stratified: bool=False) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits a DataFrame into train and test sets for cross-validation.

    Parameters:
    - df: DataFrame to be split.
    - n_splits: Number of cross-validation splits/folds.
    - stratified: Whether to perform stratified sampling. If True, the function will ensure that each split 
                  has roughly the same proportion of samples of each target class as the complete set. Assumes 
                  the target column is named 'target'.

    Returns:
    - Iterator yielding train and test DataFrames for each split.
    """
    
    # Array of indices of the dataframe
    indices = np.arange(len(df))
    
    if stratified:
        # Get target values from the DataFrame
        labels = df['target'].values
        
        # Get unique labels and their counts in the dataset
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        # Calculate the proportion of each label in the dataset
        fractions = label_counts / len(labels)
        
        test_size = int(len(df) / n_splits)
        # train_size = len(df) - test_size
        
        # Splitting for stratified cross-validation
        for fold in range(n_splits):
            test_indices = np.array([], dtype=int)
            
            # For each unique label, determine the number of instances to include in the test set
            # based on the fraction of that label in the original dataset
            for label, fraction in zip(unique_labels, fractions):
                label_indices = indices[labels == label]
                label_test_indices = np.random.choice(label_indices, size=int(test_size * fraction), replace=False)
                test_indices = np.concatenate([test_indices, label_test_indices])
                
            # The train set consists of all indices not in the test set
            train_indices = np.setdiff1d(indices, test_indices)
            
            yield df.iloc[train_indices], df.iloc[test_indices]
            
    else:
        # For non-stratified cross-validation, just shuffle the data randomly
        np.random.shuffle(indices)
        
        for fold in range(n_splits):
            # Splitting indices for train and test
            test_indices = indices[fold*int(len(indices)/n_splits):(fold+1)*int(len(indices)/n_splits)]
            train_indices = np.setdiff1d(indices, test_indices)
            
            yield df.iloc[train_indices], df.iloc[test_indices]
