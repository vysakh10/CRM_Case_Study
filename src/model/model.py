from typing import Optional, Tuple

import pandas as pd
import xgboost as xgb


def get_xgb_model(params: Optional[dict] = None) -> xgb.XGBClassifier:
    """
    Returns an XGBoost classifier with either provided or default parameters.
    Args:
        params (dict, optional): Hyperparameters for XGBoost. If None, uses defaults.
    Returns:
        xgb.XGBClassifier: Instantiated XGBoost classifier.
    """

    if params:
        xgb_best_params = params
    else:
        xgb_best_params = {
            "learning_rate": 0.006496050467463248,
            "max_depth": 5,
            "subsample": 0.9990534611257977,
            "colsample_bytree": 0.9083351644197917,
            "colsample_bylevel": 0.9455249909579699,
            "n_estimators": 298,
            "n_jobs": -1,
        }

    model = xgb.XGBClassifier(**xgb_best_params)

    return model


def train_xgb_model(
    x_train: pd.DataFrame, y_train: pd.Series, params: Optional[dict] = None
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier on the provided data.
    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (dict, optional): Hyperparameters for XGBoost. If None, uses defaults.
    Returns:
        xgb.XGBClassifier: Trained XGBoost classifier.
    """

    model = get_xgb_model(params=params)
    model.fit(x_train, y_train)

    return model


def get_xgb_predictions(
    x_test: pd.DataFrame, model: xgb.XGBClassifier
) -> Tuple[pd.Series, pd.Series]:
    """
    Generates predictions and predicted probabilities for the test set.
    Args:
        x_test (pd.DataFrame): Test features.
        model (xgb.XGBClassifier): Trained XGBoost classifier.
    Returns:
        Tuple[pd.Series, pd.Series]: Predicted labels and predicted probabilities for the positive class.
    """

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    return y_pred, y_pred_proba
