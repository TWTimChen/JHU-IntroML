# k-NN and Its Variants: A Python Implementation

This repository contains an implementation of the k-NN algorithm and its variants: Standard, Edited, and Condensed k-NN.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Training](#training)
  - [Prediction](#prediction)
- [Running the Main Script](#running-the-main-script)
- [Contributors](#contributors)
  - [Tim Chen](#tim-chen)

## Getting Started

These instructions will guide you on how to utilize the `KNN` class for your own datasets.

### Prerequisites

- Python 3.x
- NumPy
- pandas

(Additional libraries may be required based on the specific functionalities and extensions integrated into the project.)

### Installation

1. Clone the repository

    ```bash
    git clone https://github.com/TimAgro/JHU-IntroML.git
    ```

2. Navigate to the repository's directory and install the necessary packages

    ```bash
    cd lab1
    pip install -r requirements.txt
    ```

### Usage

## Initialization

To start using the k-NN model and its variants:

1. Import the KNN class from the respective module:

    ```python
    from knn import KNN
    ```

2. Instantiate the KNN class, providing the necessary hyperparameters

    ```python
    model = KNN(k=3, method="condensed", task="classification")
    ```

Parameters include:

- k: Number of neighbors.
- task: Choose between "classification" and "regression". Default is "classification".
- gamma: Parameter for regression tasks with the RBF kernel.
- epsilon: Threshold for the edited k-NN in regression tasks.
- method: Choose the k-NN variant – "standard", "edited", or "condensed". Default is "standard".

### Training

Train the model using your training data:

```python
model.fit(X_train, y_train)
```

### Prediction

After training, obtain predictions for new data:

```python
predictions = model.predict(X_test)
```

Note: Ensure that your dataset undergoes adequate preprocessing, particularly with respect to handling missing values and converting categorical features, before deploying it within the model.

## Running the Main Script

To execute the model on different datasets, run the main script:

```bash
python main.py --task 1 > result/task1.txt
python main.py --task 2 > result/task2.txt
python main.py --task 3 > result/task3.txt
python main.py --task 4 > result/task4.txt
python main.py --task 5 > result/task5.txt
python main.py --task 6 > result/task6.txt
```

## Contributors

### Tim Chen

- 📧 Email: [tchen124@jhu.edu](mailto:tchen124@jhu.edu)
- 🌐 LinkedIn: [Tim Chen](https://www.linkedin.com/in/tim-chen-017b841a9/)
- 🐱‍💻 GitHub: [@TimAgro](https://github.com/TimAgro)
