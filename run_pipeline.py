import logging

from src.config import FileArgs
from src.model.evaluation import plot_feature_importance, plot_top_k_metrics
from src.model.model import get_xgb_predictions, train_xgb_model
from src.prepare_data import get_train_test_val_split, prepare_modeling_data

logging.basicConfig(
    level=logging.INFO,  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()


def run_pipeline(args: FileArgs, feature_importance: bool = True) -> None:
    """
    Runs the end-to-end machine learning pipeline for customer modeling.

    Steps performed:
        1. Prepares modeling data using provided file arguments and feature engineering.
        2. Splits the data into training, validation, and test sets.
        3. Trains an XGBoost classifier on the training data.
        4. Generates predictions and probability scores for all data splits.
        5. Plots recall and precision at top K% for train, validation, and test sets.

    Args:
        args (FileArgs):
            An object containing file paths for customer, non-customer, and usage data.
    """

    modeling_data = prepare_modeling_data(args, feature_engineer=True)

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_val_split(
        modeling_data
    )

    logger.info("Training XGBClassifier!!")
    model = train_xgb_model(x_train=X_train, y_train=y_train)

    logger.info("Getting Predictions!!")
    y_pred_train, y_pred_probs_train = get_xgb_predictions(X_train, model)
    y_pred_val, y_pred_probs_val = get_xgb_predictions(X_val, model)
    y_pred_test, y_pred_probs_test = get_xgb_predictions(X_test, model)

    logger.info("Plotting Recall and Precision @ K")
    plot_top_k_metrics(
        y_train,
        y_pred_probs_train,
        set="Train",
        top_percent=30,
        save_path="result_charts/train_precision_recall.png",
    )

    plot_top_k_metrics(
        y_val,
        y_pred_probs_val,
        set="Val",
        top_percent=30,
        save_path="result_charts/val_precision_recall.png",
    )

    plot_top_k_metrics(
        y_test,
        y_pred_probs_test,
        set="Test",
        top_percent=30,
        save_path="result_charts/test_precision_recall.png",
    )

    logger.info("Plotting Feature Importance")
    if feature_importance:
        plot_feature_importance(
            X_train, model, save_path="result_charts/feature_importance.png"
        )


if __name__ == "__main__":
    cus_file = "dataset/customers_(4).csv"
    noncus_file = "dataset/noncustomers_(4).csv"
    usage_file = "dataset/usage_actions_(4).csv"

    args = FileArgs(
        customers_path=cus_file, non_customers_path=noncus_file, usage_path=usage_file
    )

    run_pipeline(args)
