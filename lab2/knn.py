import pandas as pd
import numpy as np


class KNN:
    def __init__(self, k, task="classification", gamma=None, epsilon=None, method="standard"):
        self.k = k
        self.task = task
        self.gamma = gamma
        self.epsilon = epsilon
        self.method = method
        self.vdm_cache = {}

    def rbf_kernel(self, x, xq, gamma):
        return np.exp(-gamma * np.linalg.norm(x-xq)**2)

    def compute_vdm(self, train_df, column, value1, value2, target_column, p=2):
        unique_classes = train_df[target_column].unique()

        # Check if this computation is already in the cache
        cache_key = (column, value1, value2)
        if cache_key in self.vdm_cache:
            return self.vdm_cache[cache_key]
        
        total = len(train_df)
        delta_sum = 0
        for c in unique_classes:
            C_i_a = len(train_df[(train_df[column] == value1) & (train_df[target_column] == c)])
            C_i = len(train_df[train_df[column] == value1])
            C_j_a = len(train_df[(train_df[column] == value2) & (train_df[target_column] == c)])
            C_j = len(train_df[train_df[column] == value2])
            
            delta_sum += abs((C_i_a / (C_i + 1e-10)) - (C_j_a / (C_j + 1e-10))) ** p
        
        result = delta_sum ** (1/p)
        
        # Store the computation in the cache
        self.vdm_cache[cache_key] = result

        return result


    def vdm_distance(self, train_df, instance1, instance2, target_column, p=2):
        distance = 0
        for col in self.one_hot_encoded_columns:
            distance += self.compute_vdm(train_df, col, instance1[col], instance2[col], target_column, p)
        return distance

    def k_nearest_neighbors(self, train_df, test_instance, k, task="classification", gamma=None, p=2):        
        # Determine columns that are not categorical (numerical)
        numerical_columns = list(set(train_df.columns) - set(self.one_hot_encoded_columns) - {"target"})

        # Converting dataframes to NumPy arrays for faster numerical computations
        train_array = train_df[numerical_columns].values.astype(float)
        test_array = test_instance[numerical_columns].values.astype(float)

        # Using vectorized operations instead of loops for numerical distance computation
        numerical_distances = np.linalg.norm(train_array - test_array, axis=1)
        
        # Initialize an empty array for categorical distances
        categorical_distances = np.zeros(len(train_df))

        # Compute categorical distances using VDM
        for index, (train_instance_name, train_instance) in enumerate(train_df.iterrows()):
            categorical_distances[index] = self.vdm_distance(train_df, test_instance, train_instance, 'target', p)

        # Calculate total distances (numerical + categorical)
        total_distances = numerical_distances + categorical_distances
        
        # Get indices of k smallest distances
        neighbors_indices = np.argpartition(total_distances, k)[:k]

        if task == "classification":
            # Plurality vote using np.bincount and np.argmax for efficiency
            targets = train_df.iloc[neighbors_indices]['target'].values
            unique, counts = np.unique(targets, return_counts=True)
            max_index = np.argmax(counts)
            return unique[max_index]
        
        elif task == "regression":
            # Weighted sum using RBF kernel
            weights = np.array([
                self.rbf_kernel(test_array, train_array[idx], gamma) for idx in neighbors_indices
            ])
            targets = train_df.iloc[neighbors_indices]['target'].values

            # Handling the case where the sum of weights is 0 to avoid division by zero
            return np.dot(weights, targets) / (np.sum(weights) + 1e-10)

    def edited_k_nearest_neighbors(self, train_df, k, task="classification", epsilon=None, gamma=None):
        original_size = train_df.shape[0]
        
        while True:
            removal_mask = []
            
            for index, row in train_df.iterrows():
                instance = row.drop("target")
                temp_train = train_df.drop(index)
                predicted = self.k_nearest_neighbors(temp_train, instance, k, task=task, gamma=gamma)
                
                # For classification
                if task == "classification" and predicted != row["target"]:
                    removal_mask.append(True)
                # For regression
                elif task == "regression" and abs(predicted - row["target"]) > epsilon:
                    removal_mask.append(True)
                else:
                    removal_mask.append(False)
            
            train_df = train_df.loc[~np.array(removal_mask)]
            
            if train_df.shape[0] == original_size or train_df.shape[0] < k+1:
                break
            
            original_size = train_df.shape[0]
        
        return train_df


    def condensed_k_nearest_neighbors(self, train_df, k, task="classification", epsilon=None, gamma=None):
        condensed_set = train_df.iloc[0:k+1, :].copy()
        train_df = train_df.drop(train_df.index[0])

        previous_size = 0
        
        while previous_size != condensed_set.shape[0]:
            previous_size = condensed_set.shape[0]
            to_add_mask = []

            for _, row in train_df.iterrows():
                instance = row.drop("target")
                predicted = self.k_nearest_neighbors(condensed_set, instance, k, task=task, gamma=gamma)
                
                # For classification
                if task == "classification" and predicted != row["target"]:
                    to_add_mask.append(True)
                # For regression
                elif task == "regression" and abs(predicted - row["target"]) > epsilon:
                    to_add_mask.append(True)
                else:
                    to_add_mask.append(False)

            to_add_mask = np.array(to_add_mask)
            condensed_set = pd.concat([condensed_set, train_df.loc[to_add_mask]])
            if np.sum(to_add_mask) > 0:
                train_df = train_df.loc[~to_add_mask]

        return condensed_set
    
    def _get_categorical(self, df):
        one_hot_encoded_columns = []
    
        for column in df.columns:
            unique_values = df[column].dropna().unique()
            if set(unique_values).issubset({0, 1}) and len(unique_values) == 2:
                one_hot_encoded_columns.append(column)
        
        return one_hot_encoded_columns


    def fit(self, X, y):
        X['target'] = y
        self.one_hot_encoded_columns = self._get_categorical(X)
        if self.method == "edited":
            self.train_df = self.edited_k_nearest_neighbors(X, self.k, self.task, self.epsilon, self.gamma)
        elif self.method == "condensed":
            self.train_df = self.condensed_k_nearest_neighbors(X, self.k, self.task, self.epsilon, self.gamma)
        else:
            self.train_df = X

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            pred = self.k_nearest_neighbors(self.train_df, row, self.k, self.task, self.gamma)
            predictions.append(pred)
        return np.array(predictions)