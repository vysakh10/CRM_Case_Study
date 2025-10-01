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
            "learning_rate": 0.0010060566063324248,
            "max_depth": 2,
            "subsample": 0.9476863439354462,
            "colsample_bytree": 0.9419907293813576,
            "colsample_bylevel": 0.9627772063758886,
            "n_estimators": 995,
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
