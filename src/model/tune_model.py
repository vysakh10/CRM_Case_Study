import logging

import optuna
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()


def precision_at_k(
    y_true: pd.Series, y_probs: pd.Series, k_percent: float = 0.1
) -> float:
    """
    Computes precision at top k% of predicted probabilities.
    Args:
        y_true (pd.Series): True binary labels.
        y_probs (pd.Series): Predicted probabilities for the positive class.
        k_percent (float, optional): Top percentage to consider. Defaults to 0.1 (10%).
    Returns:
        float: Precision at top k%.
    """

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_probs})
    df = df.sort_values("y_prob", ascending=False)
    top_k_count = max(1, int(len(df) * k_percent))
    top_k = df.head(top_k_count)
    precision = top_k["y_true"].sum() / len(top_k)

    return precision


def tune_xgb_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> dict:
    """
    Tunes hyperparameters for an XGBoost classifier using Optuna to maximize precision at top 10%.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
    Returns:
        dict: Best hyperparameters found by Optuna.
    """

    logger.warning("Hyper-parameter Tuning XGB model. This will take some time!!")

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # predict probabilities
        y_probs = model.predict_proba(X_val)[:, 1]

        # evaluate at top 10% (can make k tunable too)
        prec = precision_at_k(y_val, y_probs, k_percent=0.1)
        return prec

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Precision@10%: {study.best_trial.value}")
    logger.info(f"Best Params: {study.best_trial.params}")

    return study.best_trial.params
