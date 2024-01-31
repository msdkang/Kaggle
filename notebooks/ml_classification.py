import time
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor


def train_a_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model: object,
    verbos: bool = True,
    class_weight: dict = None,
) -> (object, dict):
    if class_weight == None:
        classes = np.unique(y_train)
        class_weight = {c: 1 / len(classes) for c in classes}

    def get_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        predicted_probabilities: np.ndarray,
    ) -> dict:
        sample_weight = [class_weight[i] for i in predicted]
        accuracy = accuracy_score(actual, predicted, sample_weight=sample_weight)
        precision = precision_score(
            actual, predicted, average="weighted", sample_weight=sample_weight
        )
        recall = recall_score(
            actual, predicted, average="weighted", sample_weight=sample_weight
        )
        f1 = f1_score(
            actual, predicted, average="weighted", sample_weight=sample_weight
        )
        auc = roc_auc_score(
            actual, predicted, average="weighted", sample_weight=sample_weight
        )
        kappa = cohen_kappa_score(actual, predicted, sample_weight=sample_weight)
        mcc = matthews_corrcoef(actual, predicted, sample_weight=sample_weight)
        logloss = log_loss(actual, predicted_probabilities, sample_weight=sample_weight)

        metrics = {
            "Accuracy": accuracy,
            "AUC": auc,
            "Recall": recall,
            "Precision": precision,
            "F1-Score": f1,
            "Kappa": kappa,
            "MCC": mcc,
            "Log Loss": logloss,
        }

        return metrics

    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
        t = time.time() - t
        y_pred = model.predict(x_train)
        y_prob = model.predict_proba(x_train)
        train_metrics = get_metrics(y_train, y_pred, y_prob)
        train_metrics["Training Time"] = t
        # do predictions
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val)
        val_metrics = get_metrics(y_val, y_pred, y_prob)
        val_metrics["Training Time"] = t

    if verbos:
        df = pd.DataFrame(
            {
                "metrics": train_metrics.keys(),
                "train": train_metrics.values(),
                "validation": val_metrics.values(),
            }
        )
        default_format = pd.options.display.float_format
        pd.options.display.float_format = "{:,.2f}".format
        print("\n*******\nResults for model {}:\n".format(model))
        display(df)
        pd.options.display.float_format = default_format

    return model, val_metrics


def compare_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    models: dict = None,
    random_state: int = 123,
) -> object:
    def sort_output(df: pd.DataFrame) -> list[int]:
        a = df.iloc[:, 1:-1].values
        a[:, :-1] = (a[:, :-1] - a[:, :-1].min(axis=0)) / (
            a[:, :-1].max(axis=0) - a[:, :-1].min(axis=0)
        )
        a[:, -1] = (a[:, -1].max(axis=0) - a[:, -1]) / (
            a[:, -1].max(axis=0) - a[:, -1].min(axis=0)
        )
        a = a.sum(axis=1)
        a = np.argsort(a)[::-1]
        return a

    def style_output(df: pd.DataFrame) -> pd.DataFrame:
        def highlight_max(s, props=""):
            return np.where(s == np.nanmax(s.values), props, "")

        def highlight_min(s, props=""):
            return np.where(s == np.nanmin(s.values), props, "")

        return df.style.apply(
            highlight_min,
            axis=0,
            props="background-color:yellow;",
            subset=df.columns.tolist()[-2:],
        ).apply(
            highlight_max,
            axis=0,
            props="background-color:yellow;",
            subset=df.columns.tolist()[1:-2],
        )

    if models == None:
        models = {
            "lr": LogisticRegression(random_state=random_state),
            "knn": KNeighborsClassifier(),
            "nb": GaussianNB(),
            "dt": DecisionTreeClassifier(random_state=random_state),
            "svm": SVC(
                kernel="linear",
                C=0.025,
                probability=True,
                random_state=random_state,
            ),
            "rbfsvm": SVC(
                gamma=2,
                C=1,
                probability=True,
                random_state=random_state,
            ),
            "mlp": MLPClassifier(random_state=random_state),
            "rf": RandomForestClassifier(random_state=random_state),
            "qda": QuadraticDiscriminantAnalysis(),
            "ada": AdaBoostClassifier(random_state=random_state),
            "gbc": GradientBoostingClassifier(random_state=random_state),
            "lda": LinearDiscriminantAnalysis(),
            "et": ExtraTreesClassifier(random_state=random_state),
            "xgboost": XGBClassifier(
                random_state=random_state,
            ),
            "lgbm": LGBMClassifier(random_state=random_state),
        }

    exp = []
    metrics = []
    N_FOLD = 5
    kfold = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=random_state)

    for key, value in models.items():
        steps = [("norm", MinMaxScaler()), ("model", value)]
        model = Pipeline(steps=steps)
        res = []
        fold = 1
        for i_in, i_out in kfold.split(x_train, y_train):
            print(
                "training a {} (fold {} of {})...".format(key, fold, N_FOLD),
                end="\r",
            )
            _, scores = train_a_model(
                x_train[i_in, :],
                y_train[i_in],
                x_train[i_out, :],
                y_train[i_out],
                model,
                verbos=False,
            )
            print(
                "{} (fold {} of {}) Accuracy = {:.2f}".format(
                    key, fold, N_FOLD, scores["Accuracy"]
                ),
                end="\r",
            )
            res.append(list(scores.values()))
            fold += 1

        print(
            key
            + "\t"
            + "".join(
                [
                    "{}: {:.2f}\t".format(k, val)
                    for k, val in zip(list(scores.keys()), np.mean(res, axis=0))
                ]
            )
        )

        metrics = ["Model"] + list(scores.keys())
        res = [key] + list(np.mean(res, axis=0))

        exp.append(res)

    exp = pd.DataFrame(exp, columns=metrics)
    sort_index = sort_output(exp)
    exp = style_output(exp.iloc[sort_index, :])
    display(exp)
    return list(models.values())[sort_index[0]]


def plot_cm(y: np.array, y_pred: np.array, labels: list[object] = ["0", "1"]) -> None:
    """plot_cm plots confusion matrix

    Parameters
    ----------
    y : np.array
        actual y
    y_pred : np.array
        predicted y
    """
    cm = confusion_matrix(y, y_pred)

    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)

    cm_display.plot()
    plt.grid(False)
    plt.show()


def plot_roc(
    y: np.array,
    y_prob: np.array,
    label: str = "model",
    class_weight: dict = None,
) -> None:
    """plot_roc _summary_

    Parameters
    ----------
    y : np.array
        actual y
    y_prob : np.array
        probability of predicted y = 1
    label : str, optional
        model name, by default "model"
    """
    if class_weight == None:
        sample_weight = None
    else:
        sample_weight = [class_weight[int(i)] for i in y]
    # roc curve for models
    fpr, tpr, thresholds = roc_curve(
        y, y_prob, pos_label=1, sample_weight=sample_weight
    )
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]
    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y))]
    p_fpr, p_tpr, _ = roc_curve(y, random_probs, pos_label=1)

    # plot roc curves
    plt.plot(fpr, tpr, linestyle="--", color="red", label=label)
    plt.plot(p_fpr, p_tpr, linestyle="--", color="blue")
    plt.title("ROC curve (Best threshold %.2f)" % best_threshold)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")

    plt.legend(loc="best")
    plt.show()


def model_evaluation(
    model: object, x: np.ndarray, y: np.array, class_weight: dict = None
) -> None:
    """model_evaluation summarizes the model evaluations

    Parameters
    ----------
    model : object
        trained model
    x : np.ndarray
        X data
    y : np.array
        y data
    """
    if class_weight == None:
        sample_weight = None
    else:
        sample_weight = [class_weight[int(i)] for i in y]
    y_pred = model.predict(x)
    print(classification_report(y, y_pred, sample_weight=sample_weight))
    plot_cm(y, y_pred)
    y_prob = model.predict_proba(x)[:, 1]
    plot_roc(y, y_prob, label=model.__class__.__name__, class_weight=class_weight)


def tune_a_classifier(
    model: Literal["rf", "xgb", "lgbm"],
    x: np.ndarray,
    y: np.array,
    sample_size: int = 1000,
    random_search: bool = True,
    random_state: int = 123,
) -> [object, object]:
    """tune_a_model tunes an random forest or xgboost hyperparameters

    Parameters
    ----------
    model : str
        model type, can be one of ["rf", "xgb", "lgbm"]
    x : np.ndarray
        x data
    y : np.array
        y data
    sample_size : int, optional
        sample size, by default 1000
    random_search : bool, optional
        a toggle to switch random or grid search, by default True

    Returns
    -------
    [object, object]
        best_estimator and best_params
    """
    if model == "rf":
        grid = {
            "n_estimators": [100, 200, 300],
            "criterion": ["gini", "entropy"],
            "max_depth": [1, 5, 10, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }
        estimator = RandomForestClassifier(random_state=random_state)
    elif model == "xgb":
        grid = {
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [1, 5, 10, 50, 100],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2, 0.3],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
            "reg_lambda": [0, 0.001, 0.01, 0.1, 1],
            "scale_pos_weight": [1, 2, 3],
            "objective": ["binary:logistic"],
            "seed": [27],
        }
        estimator = XGBClassifier(random_state=random_state)
    elif model == "lgbm":
        grid = {
            "n_estimators": [
                100,
                200,
                300,
                400,
            ],  # Number of boosting rounds or trees
            "learning_rate": [0.01, 0.1, 0.2, 0.3],  # Learning rate
            "max_depth": [
                1,
                5,
                50,
                100,
                -1,
            ],  # Maximum depth of each tree (-1 means no limit)
            "num_leaves": [15, 31, 63, 127],  # Maximum number of leaves in one tree
            "min_child_samples": [20, 50, 100],  # Minimum data in a leaf
            "subsample": [
                0.8,
                0.9,
                1.0,
            ],  # Fraction of samples used for fitting trees
            "colsample_bytree": [
                0.8,
                0.9,
                1.0,
            ],  # Fraction of features used for fitting trees
            "reg_alpha": [
                0,
                0.001,
                0.01,
                0.1,
                1,
            ],  # L1 regularization term on weights
            "reg_lambda": [
                0,
                0.001,
                0.01,
                0.1,
                1,
            ],  # L2 regularization term on weights
            "min_split_gain": [
                0.0,
                0.1,
                0.2,
                0.3,
            ],  # Minimum loss reduction required to make a further partition on a leaf node
            "boosting_type": ["gbdt", "dart", "goss"],  # Type of boosting algorithm
        }
        estimator = LGBMClassifier(random_state=random_state)
    else:
        print("Please define the model you want to tune first!")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=kfold,
    )
    if random_search:
        grid_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=grid,
            n_iter=100,
            cv=kfold,
            verbose=1,
            random_state=123,
            n_jobs=-1,
        )

    if sample_size < x.shape[0]:
        sample_id = np.random.choice(x.shape[0], sample_size, replace=False)
        X_sample, y_sample = x[sample_id, :], y[sample_id]
    else:
        X_sample, y_sample = x, y

    grid_search.fit(X_sample, y_sample)
    print(grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.best_params_


def feature_importance(
    x: np.array,
    y: np.array,
    features: list[str],
    top_k: int = 20,
    is_classifer: bool = False,
    verbos=True,
) -> pd.DataFrame:
    """
    `feature_importance` gives x and y datasets and returns feature importance

    Parameters
    ----------
    x: `np.array`
        x dataset.
    y: `np.array`
        y dataset.
    features: `list[str]`
        list of features
    top_k: `int`
        top k features, users want to have in output plot. By default is 20
    is_classifer: `bool`
        a flag to set if the problem is regression or classification. By default is `True`
    verbos: `bool`
        a flag to print results. Be default is `True`

    Returns
    -------
    importance_df: `pd.DataFrame`
        a dataframe storing features and their score.

    Notes
    -----
    """
    model = XGBRegressor()
    if is_classifer:
        model = XGBClassifier()
    model.fit(x, y)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"feature": features, "score": importance})
    importance_df.sort_values(by="score", ascending=False, inplace=True)

    if verbos:
        print("TOP {} important features".format(top_k))
        print(importance_df.head(top_k))

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.barh(
        importance_df["feature"].values[:top_k], importance_df["score"].values[:top_k]
    )
    ax.set_title("Feature Importance")
    ax.set_xlabel("score")
    ax.set_ylabel("feature")
    plt.show(block=False)
    return importance_df
