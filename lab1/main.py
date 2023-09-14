from src.loader import load_data
from src.preprocessing import DataPipeline
from src.cross_validation import train_test_split, k_fold_split
from src.evaluation_metrics import accuracy, mean_squared_error
from src.models import NullModel
import copy
import numpy as np

def printline():
    print("-"*20)

def exec_data1():
    # Load the data
    df = load_data("../assets/breast_cancer_wisconsin_original/breast-cancer-wisconsin.csv")
    print("Task 1: Breast Cancer")
    print("Type  : Classification")
    # Specify preprocessing steps
    steps = [
        {"operation": "fill_missing_values"}
    ]

    # Initialize and run the data pipeline
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)

    # rename target column to "target" for the model
    df = df.rename(columns = {"class": "target"})

    # Initialize the model
    model = NullModel(task="classification")

    # Splitting the data into (training + dev) and test datasets
    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=True)

    # Cross-validation and scoring on training set
    scores = []
    for train_fold, dev_fold in k_fold_split(train_dev_df, stratified=True):
        model.fit(train_fold.drop("target", axis=1), train_fold["target"])
        preds = model.predict(dev_fold.drop("target", axis=1))
        score = accuracy(dev_fold["target"], preds)
        scores.append(score)

    # Calculating average accuracy over all the folds
    avg_accuracy = sum(scores) / len(scores)
    print(f"Average accuracy over all the folds: {avg_accuracy:.2%}", )

    # evaluating on the test set
    model.fit(train_dev_df.drop("target", axis=1), train_dev_df["target"])
    test_preds = model.predict(test_df.drop("target", axis=1))
    test_score = accuracy(test_df["target"], test_preds)
    print(f"Accuracy on the test set: {test_score:.2%}")

def exec_data2():
    # Load the data
    df = load_data("../assets/car_evaluation/car.csv")
    print("Task 2: Car Evaluation")
    print("Type  : Classification")
    # Specify preprocessing steps
    steps = [
        {"operation": "fill_missing_values"},
        {"operation": "handle_ordinal_data","params": {"column": "buying", "mapping": {"v-high": 4, "high": 3, "med": 2, "low": 1}}},
        {"operation": "handle_ordinal_data","params": {"column": "maint", "mapping": {"v-high": 4, "high": 3, "med": 2, "low": 1}}},
        {"operation": "handle_ordinal_data","params": {"column": "doors", "mapping": {"2": 2, "3": 3, "4": 4, "5-more": 6}}},
        {"operation": "handle_ordinal_data","params": {"column": "persons", "mapping": {"2": 2, "4": 4, "more": 6}}},
        {"operation": "handle_ordinal_data","params": {"column": "lug_boot", "mapping": {"small": 0, "med": 1, "big": 2}}},
        {"operation": "handle_ordinal_data","params": {"column": "safety", "mapping": {"low": 0, "med": 1, "high": 2}}},
        
    ]

    # Initialize and run the data pipeline
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)

    # rename target column to "target" for the model
    df = df.rename(columns = {"class": "target"})

    # Initialize the model
    model = NullModel(task="classification")

    # Splitting the data into (training + dev) and test datasets
    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=True)

    # Cross-validation and scoring on training set
    scores = []
    for train_fold, dev_fold in k_fold_split(train_dev_df, stratified=True):
        model.fit(train_fold.drop("target", axis=1), train_fold["target"])
        preds = model.predict(dev_fold.drop("target", axis=1))
        score = accuracy(dev_fold["target"], preds)
        scores.append(score)

    # Calculating average accuracy over all the folds
    avg_accuracy = sum(scores) / len(scores)
    print(f"Average accuracy over all the folds: {avg_accuracy:.2%}", )

    # evaluating on the test set
    model.fit(train_dev_df.drop("target", axis=1), train_dev_df["target"])
    test_preds = model.predict(test_df.drop("target", axis=1))
    test_score = accuracy(test_df["target"], test_preds)
    print(f"Accuracy on the test set: {test_score:.2%}")

def exec_data3():
    # Load the data
    df = load_data("../assets/congressional_voting_records/house-votes-84.csv")
    print("Task 3: Congressional Vote")
    print("Type  : Classification")
    # Specify preprocessing steps
    steps = [
        {"operation": "fill_missing_values"},
    ]
    template = {"operation": "one_hot_encode","params": {"column": ""}}
    for c in list(df.columns[1:]):
        new_step = copy.deepcopy(template)
        new_step["params"]["column"] = c
        steps.append(new_step)

    # Initialize and run the data pipeline
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)

    # rename target column to "target" for the model
    df = df.rename(columns = {"class": "target"})

    # Initialize the model
    model = NullModel(task="classification")

    # Splitting the data into (training + dev) and test datasets
    train_dev_df, test_df = train_test_split(df, test_size=0.2, stratified=True)

    # Cross-validation and scoring on training set
    scores = []
    for train_fold, dev_fold in k_fold_split(train_dev_df, stratified=True):
        model.fit(train_fold.drop("target", axis=1), train_fold["target"])
        preds = model.predict(dev_fold.drop("target", axis=1))
        score = accuracy(dev_fold["target"], preds)
        scores.append(score)

    # Calculating average accuracy over all the folds
    avg_accuracy = sum(scores) / len(scores)
    print(f"Average accuracy over all the folds: {avg_accuracy:.2%}", )

    # evaluating on the test set
    model.fit(train_dev_df.drop("target", axis=1), train_dev_df["target"])
    test_preds = model.predict(test_df.drop("target", axis=1))
    test_score = accuracy(test_df["target"], test_preds)
    print(f"Accuracy on the test set: {test_score:.2%}")

def exec_data4():
    # Load the data
    df = load_data("../assets/abalone/abalone.csv")
    print("Task 4: Abalone")
    print("Type  : Regression")
    # Specify preprocessing steps
    steps = [
        {"operation": "one_hot_encode","params": {"column": "sex"}},
        {"operation": "standardize", "params": {"column": "height"}}
    ]

    # Initialize and run the data pipeline
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)

    # rename target column to "target" for the model
    df = df.rename(columns = {"rings": "target"})

    # Initialize the model
    model = NullModel(task="regression")

    # Splitting the data into (training + dev) and test datasets
    train_dev_df, test_df = train_test_split(df, test_size=0.2)

    # Cross-validation and scoring on training set
    scores = []
    for train_fold, dev_fold in k_fold_split(train_dev_df):
        model.fit(train_fold.drop("target", axis=1), train_fold["target"])
        preds = model.predict(dev_fold.drop("target", axis=1))
        score = mean_squared_error(dev_fold["target"], preds)
        scores.append(score)

    # Calculating average MSE over all the folds
    avg_mse = sum(scores) / len(scores)
    print(f"Average MSE over all the folds: {avg_mse:10.4f}", )

    # evaluating on the test set
    model.fit(train_dev_df.drop("target", axis=1), train_dev_df["target"])
    test_preds = model.predict(test_df.drop("target", axis=1))
    test_score = mean_squared_error(test_df["target"], test_preds)
    print(f"MSE on the test set: {test_score:10.4f}")

def exec_data5():
    # Load the data
    df = load_data("../assets/computer_hardware/machine.csv")
    print("Task 5: Computer Hardware")
    print("Type  : Regression")
    # Specify preprocessing steps
    steps = [
        {"operation": "one_hot_encode","params": {"column": "vendor_name"}},
        {"operation": "one_hot_encode","params": {"column": "model_name"}},
        {"operation": "standardize", "params": {"column": "MYCT"}}
    ]

    # Initialize and run the data pipeline
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)

    # rename target column to "target" for the model
    df = df.rename(columns = {"PRP": "target"})

    # Initialize the model
    model = NullModel(task="regression")

    # Splitting the data into (training + dev) and test datasets
    train_dev_df, test_df = train_test_split(df, test_size=0.2)

    # Cross-validation and scoring on training set
    scores = []
    for train_fold, dev_fold in k_fold_split(train_dev_df):
        model.fit(train_fold.drop("target", axis=1), train_fold["target"])
        preds = model.predict(dev_fold.drop("target", axis=1))
        score = mean_squared_error(dev_fold["target"], preds)
        scores.append(score)

    # Calculating average MSE over all the folds
    avg_mse = sum(scores) / len(scores)
    print(f"Average MSE over all the folds: {avg_mse:10.4f}", )

    # evaluating on the test set
    model.fit(train_dev_df.drop("target", axis=1), train_dev_df["target"])
    test_preds = model.predict(test_df.drop("target", axis=1))
    test_score = mean_squared_error(test_df["target"], test_preds)
    print(f"MSE on the test set: {test_score:10.4f}")

def exec_data6():
    # Load the data
    df = load_data("../assets/forest_fires/forestfires.csv")
    print("Task 6: Forest Fires")
    print("Type  : Regression")
    # Specify preprocessing steps
    steps = [
        {"operation": "one_hot_encode","params": {"column": "month"}},
        {"operation": "one_hot_encode","params": {"column": "day"}},
        {"operation": "standardize", "params": {"column": "FFMC"}},
        {"operation": "function_fransformer", "params": {"column": "area", "func":lambda x: np.log(x + 1)}}
    ]

    # Initialize and run the data pipeline
    pipeline = DataPipeline(steps)
    df = pipeline.run(df)

    # rename target column to "target" for the model
    df = df.rename(columns = {"area": "target"})

    # Initialize the model
    model = NullModel(task="regression")

    # Splitting the data into (training + dev) and test datasets
    train_dev_df, test_df = train_test_split(df, test_size=0.2)

    # Cross-validation and scoring on training set
    scores = []
    for train_fold, dev_fold in k_fold_split(train_dev_df):
        model.fit(train_fold.drop("target", axis=1), train_fold["target"])
        preds = model.predict(dev_fold.drop("target", axis=1))
        score = mean_squared_error(dev_fold["target"], preds)
        scores.append(score)

    # Calculating average MSE over all the folds
    avg_mse = sum(scores) / len(scores)
    print(f"Average MSE over all the folds: {avg_mse:10.4f}", )

    # evaluating on the test set
    model.fit(train_dev_df.drop("target", axis=1), train_dev_df["target"])
    test_preds = model.predict(test_df.drop("target", axis=1))
    test_score = mean_squared_error(test_df["target"], test_preds)
    print(f"MSE on the test set: {test_score:10.4f}")

if __name__ == "__main__":
    exec_data1()
    printline()
    exec_data2()
    printline()
    exec_data3()
    printline()
    exec_data4()
    printline()
    exec_data5()
    printline()
    exec_data6()
    printline()
