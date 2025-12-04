from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


def kfold_random_forest(X, y):
    # NB: adjust search space and n_splits if appropriate.
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    clf = _grid_search_random_forest(train_X, train_y)
    metrics = _get_metrics(clf, test_X, test_y)
    return clf.best_estimator_, metrics


def _grid_search_random_forest(train_X, train_y):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    model = RandomForestClassifier(random_state=0)
    space = dict()
    space["n_estimators"] = [i for i in range(1, 11)]
    space["max_depth"] = [i for i in range(1, 11)]
    search = GridSearchCV(model, space, scoring="accuracy", cv=kf)
    return search.fit(train_X, train_y)


def _get_metrics(clf, test_X, test_y):
    pred_y = clf.predict(test_X)
    metrics = dict()
    metrics["accuracy"] = accuracy_score(test_y, pred_y)
    metrics["f1"] = f1_score(test_y, pred_y)
    metrics["roc_auc"] = roc_auc_score(test_y, pred_y)
    return metrics
