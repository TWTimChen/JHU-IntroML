import numpy as np
import sys
import argparse
from tqdm import tqdm

sys.path.append("../lab1")
from mltk.loader import load_data
from mltk.preprocessing import DataPipeline
from mltk.cross_validation import train_test_split, k_fold_split
from mltk.evaluation_metrics import accuracy, accuracy_with_error
from knn import KNN


def evaluate_model(df, model, n_splits, metrics):
    # Splitting the data into (training + dev) and test datasets
    scores = []
    for train_fold, dev_fold in k_fold_split(df, n_splits=n_splits, stratified=True if model.task == "classification" else False):
        model.fit(train_fold.drop("target", axis=1), train_fold["target"])
        preds = model.predict(dev_fold.drop("target", axis=1))
        if model.task == "classification":
            score = metrics(dev_fold["target"], preds)
        else:
            score = metrics(dev_fold["target"], preds, model.epsilon)
        scores.append(score)
    return np.array(scores)

def grid_search(df, param_grid, task, method, metrics):
    best_params = {}
    best_score = float('-inf')  # we'll use MSE as the metric, so initialize with a high score

    print(f"Performing grid search for {method} k-NN...")
    param_combinations = []

    for k in param_grid['k']:
        for epsilon in param_grid['epsilon']:
            for gamma in param_grid['gamma']:
                param_combinations.append((k, epsilon, gamma))

    for k, epsilon, gamma in tqdm(param_combinations):
        model = KNN(k=k, task=task, gamma=gamma, epsilon=epsilon, method=method)
        
        current_score = evaluate_model(df, model, n_splits=5, metrics=metrics)
        current_score = np.mean(current_score)
        
        if current_score > best_score:
            best_score = current_score
            best_params = {'k': k, 'epsilon': epsilon, 'gamma': gamma}

    return best_params, best_score

def run_search(train_dev_df, param_grid, task, method, model_name, param_df, test_df, metrics, n_splits=5, n_repeats=5):
    # Grid Search
    best_params, best_score = grid_search(param_df, param_grid, task=task, method=method, metrics=metrics)
    print(f"Best {model_name} params: {best_params}")
    print(f"Best {model_name} score: {best_score:.2f}")
    
    # Model Initialization and Evaluation
    model = KNN(k=best_params['k'], task=task, gamma=best_params['gamma'], epsilon=best_params['epsilon'], method=method)
    scores = [evaluate_model(train_dev_df, model, n_splits=n_splits, metrics=metrics) for _ in range(n_repeats)]
    
    # Display Results
    print(f"Average {model_name} score: {np.mean(scores):.2f}")
    print(f"Standard deviation of {model_name} score: {np.std(scores):.4f}")
    if model.task == "classification":
        print(f"Test score: {metrics(test_df['target'], model.predict(test_df.drop('target', axis=1))):.2f}")
    else:
        print(f"Test score: {metrics(test_df['target'], model.predict(test_df.drop('target', axis=1)), model.epsilon):.2f}")
    
    return model

def exec_data1():
    # Load the data
    df = load_data("../assets/breast_cancer_wisconsin_original/breast-cancer-wisconsin.csv")
    df = df.drop(['sample_code_number'], axis=1)

    # Specify preprocessing steps
    steps = [
        {"operation": "fill_missing_values"},
        {"operation": "standardize", "params": {"column": [
            "clump_thickness", "uni_cell_size", "uni_cell_shape",
            "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
            "bland_chromatin", "normal_nucleoli", "mitoses"
            ]}
        },
    ]
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)
    df = df.rename(columns={"class": "target"})  # rename target column for the sake of consistency

    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=True)
    train_dev_df, param_df = train_test_split(train_dev_df, test_size=0.25, stratified=True)

    # Define the hyperparameter search space
    param_grid = {
        'k': [1, 3, 5 ,7, 9],  # Possible values for k
        'epsilon': [None],
        'gamma': [None],
    }

    # Evaluations
    run_search(train_dev_df, param_grid, "classification", "standard", "k-NN", param_df, test_df, accuracy)
    run_search(train_dev_df, param_grid, "classification", "edited", "Edited k-NN", param_df, test_df, accuracy)
    run_search(train_dev_df, param_grid, "classification", "condensed", "Condensed k-NN", param_df, test_df, accuracy)

def exec_data2():
    # Load the data
    df = load_data("../assets/car_evaluation/car.csv")
    steps = [
        {"operation": "fill_missing_values"},
        {"operation": "handle_ordinal_data","params": {"column": "buying", "mapping": {"vhigh": 4, "high": 3, "med": 2, "low": 1}}},
        {"operation": "handle_ordinal_data","params": {"column": "maint", "mapping": {"vhigh": 4, "high": 3, "med": 2, "low": 1}}},
        {"operation": "handle_ordinal_data","params": {"column": "doors", "mapping": {"2": 2, "3": 3, "4": 4, "5more": 6}}},
        {"operation": "handle_ordinal_data","params": {"column": "persons", "mapping": {"2": 2, "4": 4, "more": 6}}},
        {"operation": "handle_ordinal_data","params": {"column": "lug_boot", "mapping": {"small": 0, "med": 1, "big": 2}}},
        {"operation": "handle_ordinal_data","params": {"column": "safety", "mapping": {"low": 0, "med": 1, "high": 2}}},
    ]
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)
    df = df.rename(columns={"class": "target"})  # rename target column for the sake of consistency

    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=True)
    train_dev_df, param_df = train_test_split(train_dev_df, test_size=0.25, stratified=True)

    # Define the hyperparameter search space
    param_grid = {
        'k': [1, 3, 5 ,7, 9],  # Possible values for k
        'epsilon': [None],
        'gamma': [None],
    }

    # Evaluations
    run_search(train_dev_df, param_grid, "classification", "standard", "k-NN", param_df, test_df, accuracy)
    run_search(train_dev_df, param_grid, "classification", "edited", "Edited k-NN", param_df, test_df, accuracy)
    run_search(train_dev_df, param_grid, "classification", "condensed", "Condensed k-NN", param_df, test_df, accuracy)

def exec_data3():
    # Load the data
    df = load_data("../assets/congressional_voting_records/house-votes-84.csv")
    steps = [
        {"operation": "fill_missing_values"},
    ]
    for c in df.columns[1:]:
        steps.append({"operation": "one_hot_encode", "params": {"column": c}})
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)
    df = df.rename(columns={"class": "target"})  # rename target column for the sake of consistency

    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=True)
    train_dev_df, param_df = train_test_split(train_dev_df, test_size=0.25, stratified=True)

    # Define the hyperparameter search space
    param_grid = {
        'k': [1, 3, 5 ,7, 9],  # Possible values for k
        'epsilon': [None],
        'gamma': [None],
    }

    # Evaluations
    run_search(train_dev_df, param_grid, "classification", "standard", "k-NN", param_df, test_df, accuracy)
    run_search(train_dev_df, param_grid, "classification", "edited", "Edited k-NN", param_df, test_df, accuracy)
    run_search(train_dev_df, param_grid, "classification", "condensed", "Condensed k-NN", param_df, test_df, accuracy)

def exec_data4():
    # Load the data
    df = load_data("../assets/abalone/abalone.csv")
    steps = [
        {"operation": "fill_missing_values"},
        {"operation": "one_hot_encode","params": {"column": "sex"}},
        {"operation": "standardize", "params": {"column": ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]}},
    ]
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)
    df = df.rename(columns={"rings": "target"})  # rename target column for the sake of consistency

    _, df = train_test_split(df, test_size=0.4, stratified=False)
    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=False)
    train_dev_df, param_df = train_test_split(train_dev_df, test_size=0.25, stratified=False)

    # Define the hyperparameter search space
    param_grid = {
        'k': [3, 5 ,7],  # Possible values for k
        'epsilon': [1],  # Possible values for epsilon
        'gamma': [0.1, 1, 5]  # Possible values for gamma
    }

    # Evaluations
    run_search(train_dev_df, param_grid, "regression", "standard", "k-NN", param_df, test_df, accuracy_with_error)
    run_search(train_dev_df, param_grid, "regression", "edited", "Edited k-NN", param_df, test_df, accuracy_with_error)
    run_search(train_dev_df, param_grid, "regression", "condensed", "Condensed k-NN", param_df, test_df, accuracy_with_error)

def exec_data5():
    # Load the data
    df = load_data("../assets/computer_hardware/machine.csv")
    steps = [
        {"operation": "fill_missing_values"},
        {"operation": "one_hot_encode","params": {"column": "vendor_name"}},
        {"operation": "one_hot_encode","params": {"column": "model_name"}},
        {"operation": "standardize", "params": {"column": ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]}},
    ]

    pipeline = DataPipeline(steps)
    df = pipeline.run(df)
    df = df.rename(columns={"ERP": "target"})  # rename target column for the sake of consistency

    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=False)
    train_dev_df, param_df = train_test_split(train_dev_df, test_size=0.25, stratified=False)

    # Define the hyperparameter search space
    param_grid = {
        'k': [1, 3, 5 ,7, 9],  # Possible values for k
        'epsilon': [30],  # Possible values for epsilon
        'gamma': [0.1, 1, 5]  # Possible values for gamma
    }

    # Evaluations
    run_search(train_dev_df, param_grid, "regression", "standard", "k-NN", param_df, test_df, accuracy_with_error)
    run_search(train_dev_df, param_grid, "regression", "edited", "Edited k-NN", param_df, test_df, accuracy_with_error)
    run_search(train_dev_df, param_grid, "regression", "condensed", "Condensed k-NN", param_df, test_df, accuracy_with_error)

def exec_data6():
    # Load the data
    df = load_data("../assets/forest_fires/forestfires.csv")
    steps = [
        {"operation": "fill_missing_values"},
        {"operation": "one_hot_encode","params": {"column": "month"}},
        {"operation": "one_hot_encode","params": {"column": "day"}},
        {"operation": "standardize", "params": {"column": ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]}},
    ]

    pipeline = DataPipeline(steps)
    df = pipeline.run(df)
    df = df.rename(columns={"area": "target"})  # rename target column for the sake of consistency

    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=False)
    train_dev_df, param_df = train_test_split(train_dev_df, test_size=0.25, stratified=False)

    # Define the hyperparameter search space
    param_grid = {
        'k': [1, 3, 5 ,7, 9],  # Possible values for k
        'epsilon': [3],  # Possible values for epsilon
        'gamma': [0.1, 1, 5]  # Possible values for gamma
    }

    # Evaluations
    run_search(train_dev_df, param_grid, "regression", "standard", "k-NN", param_df, test_df, accuracy_with_error)
    run_search(train_dev_df, param_grid, "regression", "edited", "Edited k-NN", param_df, test_df, accuracy_with_error)
    run_search(train_dev_df, param_grid, "regression", "condensed", "Condensed k-NN", param_df, test_df, accuracy_with_error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, help="Task number", required=True)
    args = parser.parse_args()
    if args.task == 1:
        exec_data1()
    elif args.task == 2:
        exec_data2()
    elif args.task == 3:
        exec_data3()
    elif args.task == 4:
        exec_data4()
    elif args.task == 5:
        exec_data5()
    elif args.task == 6:
        exec_data6()
