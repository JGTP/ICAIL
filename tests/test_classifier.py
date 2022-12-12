from classifier import kfold_random_forest
from preprocessing import get_data


def test_model_pipeline():
    data = get_data("admission")
    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    clf, metrics = kfold_random_forest(X, y)
    assert clf._estimator_type == "classifier"
    assert round(metrics["accuracy"], 2) == 0.94
    assert round(metrics["f1"], 2) == 0.97
    assert round(metrics["roc_auc"], 2) == 0.66
