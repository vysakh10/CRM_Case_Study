import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FileArgs
from src.feature_engineering import get_features, standard_scale_num_features

logging.basicConfig(
    level=logging.INFO,  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()


def prepare_weekly_usage_data_for_everyone(
    fileargs: FileArgs, output_file_path: str = None
) -> pd.DataFrame:
    """
    Combines customer, non-customer, and usage action files to create the final weekly report data.

    Args:
        fileargs (FileArgs): Object containing file paths for customers, non-customers, and usage data.
        output_file_path (str, optional): If provided, saves the resulting DataFrame to this CSV file.

    Returns:
        pd.DataFrame: Combined and processed weekly usage data for all users.
    """

    customers_df = pd.read_csv(fileargs.customers_path)
    noncustomers_df = pd.read_csv(fileargs.non_customers_path)
    usage_actions_df = pd.read_csv(fileargs.usage_path)

    # create a flag for train and test set
    # customers data will be used for training
    customers_df["customers"] = "yes"
    noncustomers_df["customers"] = "no"

    metadata = pd.concat([customers_df, noncustomers_df], ignore_index=True)

    metadata.drop_duplicates(subset=["id"], inplace=True)

    complete_report_data = pd.merge(
        usage_actions_df, metadata, how="left", on="id"
    ).sort_values(by="WHEN_TIMESTAMP")

    complete_report_data["WHEN_TIMESTAMP"] = pd.to_datetime(
        complete_report_data["WHEN_TIMESTAMP"]
    )

    if output_file_path:
        complete_report_data.to_csv(output_file_path, index=False)

    return complete_report_data


def prepare_modeling_data(
    fileargs: FileArgs,
    feature_engineer: bool = True,
    output_file_path: str = None,
    modeling: bool = True,
) -> pd.DataFrame:
    """
    Prepares the modeling dataset by merging customer and usage data, engineering features, and labeling.

    Args:
        fileargs (FileArgs): Object containing file paths for customers and usage data.
        feature_engineer (bool, optional): Whether to apply feature engineering. Defaults to True.
        output_file_path (str, optional): If provided, saves the resulting DataFrame to this CSV file.
        modeling (bool, optional): If True, prepares data for modeling; if False, for EDA. Defaults to True.

    Returns:
        pd.DataFrame: The processed modeling data ready for training or EDA.
    """
    customers_df = pd.read_csv(fileargs.customers_path)
    usage_actions_df = pd.read_csv(fileargs.usage_path)

    customers_df["CLOSEDATE"] = pd.to_datetime(customers_df["CLOSEDATE"])
    usage_actions_df["WHEN_TIMESTAMP"] = pd.to_datetime(
        usage_actions_df["WHEN_TIMESTAMP"]
    )

    customers_df.drop_duplicates(
        subset=["id"], inplace=True
    )  # drop any duplicates if present

    modeling_df = pd.merge(
        usage_actions_df,
        customers_df[["id", "CLOSEDATE", "ALEXA_RANK", "EMPLOYEE_RANGE", "INDUSTRY"]],
        how="left",
        on="id",
    )

    modeling_df["DAYS_DIFF"] = (
        modeling_df["CLOSEDATE"] - modeling_df["WHEN_TIMESTAMP"]
    ).dt.days

    modeling_df = modeling_df[
        modeling_df["DAYS_DIFF"] > 0
    ]  # select rows before conversion date

    modeling_df["WILL_CONVERT"] = (modeling_df["DAYS_DIFF"] <= 30).astype(int)

    if feature_engineer:
        modeling_df = get_features(modeling_df, modeling)

    if modeling:
        modeling_df.drop(["CLOSEDATE", "DAYS_DIFF"], axis=1, inplace=True)

    if output_file_path:
        modeling_df.to_csv(output_file_path, index=False)

    return modeling_df


def get_train_test_val_split(
    modeling_data: pd.DataFrame,
    scale_num_features: bool = True,
    train_perc: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits the modeling data into train, validation, and test sets, with optional scaling of numeric features.

    Args:
        modeling_data (pd.DataFrame): The full modeling dataset.
        scale_num_features (bool, optional): Whether to standard scale numeric features. Defaults to True.
        train_perc (float, optional): Proportion of data to use for training. Defaults to 0.7.

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Training features
            - X_val (pd.DataFrame): Validation features
            - X_test (pd.DataFrame): Test features
            - y_train (pd.Series): Training labels
            - y_val (pd.Series): Validation labels
            - y_test (pd.Series): Test labels
    """

    # Extract unique IDs
    unique_ids = modeling_data["id"].unique()

    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=1 - train_perc, random_state=42
    )

    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=23)

    # Create datasets based on IDs
    train_df = modeling_data[modeling_data["id"].isin(train_ids)].copy()
    val_df = modeling_data[modeling_data["id"].isin(val_ids)].copy()
    test_df = modeling_data[modeling_data["id"].isin(test_ids)].copy()

    X_train = train_df.drop(columns=["WILL_CONVERT", "id"])
    y_train = train_df["WILL_CONVERT"]

    X_val = val_df.drop(columns=["WILL_CONVERT", "id"])
    y_val = val_df["WILL_CONVERT"]

    X_test = test_df.drop(columns=["WILL_CONVERT", "id"])
    y_test = test_df["WILL_CONVERT"]

    logger.info(f"Train Shape: {X_train.shape, y_train.shape}")
    logger.info(f"Validation Shape: {X_val.shape, y_val.shape}")
    logger.info(f"Test Shape: {X_test.shape, y_test.shape}")

    if scale_num_features:
        x_train_scaled, x_val_scaled, x_test_scaled = standard_scale_num_features(
            X_train, X_val, X_test
        )

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test
