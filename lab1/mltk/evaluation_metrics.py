import numpy as np

class ListComparison:
    def __init__(self, data):
        self.data = data
    
    def __eq__(self, other):
        if not isinstance(other, ListComparison):
            return NotImplemented
        
        if len(self.data) != len(other.data):
            raise ValueError("Both lists must have the same length for comparison.")
        
        return [a == b for a, b in zip(self.data, other.data)]


def pairwise_comparison(func):
    def wrapper(y_true, y_pred, *args, **kwargs):
        try:
            y_true, y_pred = np.array(y_true), np.array(y_pred)
        except:
            # Wrap lists in the custom class
            y_true, y_pred = ListComparison(y_true), ListComparison(y_pred)
        return func(y_true, y_pred, *args, **kwargs)
    return wrapper

# Classification Metrics
@pairwise_comparison
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def _get_classes(y_true):
    classes = np.unique(y_true)
    if len(classes) != 2:
        print(y_true)
        raise ValueError("y_true is not binary")
    return classes[0], classes[1]

@pairwise_comparison
def precision(y_true, y_pred):
    neg_class, pos_class = _get_classes(y_true)
    tp = np.sum((y_true == pos_class) & (y_pred == pos_class))
    fp = np.sum((y_true == neg_class) & (y_pred == pos_class))
    denominator = tp + fp
    if denominator == 0:
        return 0
    return tp / denominator

@pairwise_comparison
def recall(y_true, y_pred):
    neg_class, pos_class = _get_classes(y_true)
    tp = np.sum((y_true == pos_class) & (y_pred == pos_class))
    fn = np.sum((y_true == pos_class) & (y_pred == neg_class))
    denominator = tp + fn
    if denominator == 0:
        return 0
    return tp / denominator

@pairwise_comparison
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)


# Regression Metrics
@pairwise_comparison
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

@pairwise_comparison
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

@pairwise_comparison
def r2_coefficient(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)

@pairwise_comparison
def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
