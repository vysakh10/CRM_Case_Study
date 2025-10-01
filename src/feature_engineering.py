import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import NUM_COLS, employee_range_map


def cum_actions_contacts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for ACTIONS_CRM_CONTACTS per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'ACTIONS_CRM_CONTACTS'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_ACTIONS_CONTACTS' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_ACTIONS_CONTACTS"] = data.groupby("id")[
        "ACTIONS_CRM_CONTACTS"
    ].cumsum()

    return data


def cum_actions_companies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for ACTIONS_CRM_COMPANIES per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'ACTIONS_CRM_COMPANIES'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_ACTIONS_COMPANIES' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_ACTIONS_COMPANIES"] = data.groupby("id")[
        "ACTIONS_CRM_COMPANIES"
    ].cumsum()

    return data


def cum_actions_deals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for ACTIONS_CRM_DEALS per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'ACTIONS_CRM_DEALS'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_ACTIONS_DEALS' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_ACTIONS_DEALS"] = data.groupby("id")["ACTIONS_CRM_DEALS"].cumsum()

    return data


def cum_actions_emails(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for ACTIONS_EMAIL per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'ACTIONS_EMAIL'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_ACTIONS_EMAILS' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_ACTIONS_EMAILS"] = data.groupby("id")["ACTIONS_EMAIL"].cumsum()

    return data


def cum_users_contacts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for USERS_CRM_CONTACTS per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'USERS_CRM_CONTACTS'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_USERS_CONTACTS' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_USERS_CONTACTS"] = data.groupby("id")[
        "USERS_CRM_CONTACTS"
    ].cumsum()

    return data


def cum_users_companies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for USERS_CRM_COMPANIES per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'USERS_CRM_COMPANIES'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_USERS_COMPANIES' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_USERS_COMPANIES"] = data.groupby("id")[
        "USERS_CRM_COMPANIES"
    ].cumsum()

    return data


def cum_users_deals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for USERS_CRM_DEALS per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'USERS_CRM_DEALS'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_USERS_DEALS' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_USERS_DEALS"] = data.groupby("id")["USERS_CRM_DEALS"].cumsum()

    return data


def cum_users_emails(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a cumulative sum column for USERS_EMAIL per user over time.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'USERS_EMAIL'.
    Returns:
        pd.DataFrame: DataFrame with 'CUMULATIVE_USERS_EMAILS' column added.
    """

    data = data.sort_values(by=["id", "WHEN_TIMESTAMP"])
    data["CUMULATIVE_USERS_EMAILS"] = data.groupby("id")["USERS_EMAIL"].cumsum()

    return data


def get_date_features(
    data: pd.DataFrame, col: str, modeling: bool = True
) -> pd.DataFrame:
    """
    Extracts day, month, and year from a datetime column. Optionally drops the timestamp column for modeling.
    Args:
        data (pd.DataFrame): Input DataFrame.
        col (str): Name of the datetime column.
        modeling (bool, optional): If True, drops the timestamp column. Defaults to True.
    Returns:
        pd.DataFrame: DataFrame with new date features.
    """

    data[col] = pd.to_datetime(data[col])
    data["DAY"] = data[col].dt.day
    data["MONTH"] = data[col].dt.month
    data["YEAR"] = data[col].dt.year
    if modeling:
        data = data.drop(["WHEN_TIMESTAMP"], axis=1)

    return data


def rem_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers based on hardcoded thresholds for several columns.
    Args:
        data (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """

    data = data[data["ACTIONS_CRM_CONTACTS"] < 3000]
    data = data[data["ACTIONS_CRM_COMPANIES"] < 1000]
    data = data[data["ACTIONS_CRM_DEALS"] < 1000]
    data = data[data["USERS_CRM_COMPANIES"] < 30]
    data = data[data["USERS_CRM_DEALS"] < 40]
    data = data[data["ACTIONS_EMAIL"] < 175]

    return data


def fill_na_employee_range(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing EMPLOYEE_RANGE values based on ALEXA_RANK.
    Args:
        data (pd.DataFrame): Input DataFrame with 'EMPLOYEE_RANGE' and 'ALEXA_RANK'.
    Returns:
        pd.DataFrame: DataFrame with missing EMPLOYEE_RANGE filled.
    """

    if data[data["EMPLOYEE_RANGE"].isna()]["ALEXA_RANK"].values[0] > 15000001.0:
        data["EMPLOYEE_RANGE"] = data["EMPLOYEE_RANGE"].fillna("1")

    return data


def log_scale(data: pd.DataFrame, col: str, modeling: bool = True) -> pd.DataFrame:
    """
    Applies log1p scaling to a column and optionally drops the original columns for modeling.
    Args:
        data (pd.DataFrame): Input DataFrame.
        col (str): Name of the column to log scale.
        modeling (bool, optional): If True, drops original columns. Defaults to True.
    Returns:
        pd.DataFrame: DataFrame with log-scaled column.
    """

    data["ALEXA_RANK_CLEAN"] = data["ALEXA_RANK"].replace(0, np.nan)
    data["ALEXA_RANK_LOG"] = np.log1p(data["ALEXA_RANK_CLEAN"])

    if modeling:
        data = data.drop(["ALEXA_RANK", "ALEXA_RANK_CLEAN"], axis=1)

    return data


def cyclic_encode_day(data: pd.DataFrame, modeling: bool = True) -> pd.DataFrame:
    """
    Adds a sine-encoded feature for the day of the month. Optionally drops the original column.
    Args:
        data (pd.DataFrame): Input DataFrame with 'DAY'.
        modeling (bool, optional): If True, drops 'DAY'. Defaults to True.
    Returns:
        pd.DataFrame: DataFrame with sine-encoded day feature.
    """

    data["day_sin"] = np.sin(2 * np.pi * data["DAY"] / 31)

    if modeling:
        data = data.drop(["DAY"], axis=1)

    return data


def cyclic_encode_month(data: pd.DataFrame, modeling: bool = True) -> pd.DataFrame:
    """
    Adds a sine-encoded feature for the month. Optionally drops the original column.
    Args:
        data (pd.DataFrame): Input DataFrame with 'MONTH'.
        modeling (bool, optional): If True, drops 'MONTH'. Defaults to True.
    Returns:
        pd.DataFrame: DataFrame with sine-encoded month feature.
    """

    data["month_sin"] = np.sin(2 * np.pi * data["MONTH"] / 12)

    if modeling:
        data = data.drop(["MONTH"], axis=1)

    return data


def map_employee_range(data: pd.DataFrame, modeling: bool = True) -> pd.DataFrame:
    """
    Maps EMPLOYEE_RANGE to a numeric value and optionally drops the original column.
    Args:
        data (pd.DataFrame): Input DataFrame with 'EMPLOYEE_RANGE'.
        modeling (bool, optional): If True, drops 'EMPLOYEE_RANGE'. Defaults to True.
    Returns:
        pd.DataFrame: DataFrame with mapped employee range.
    """

    data["EMPLOYEE_RANGE_MAPPED"] = data["EMPLOYEE_RANGE"].map(employee_range_map)

    if modeling:
        data = data.drop(["EMPLOYEE_RANGE"], axis=1)

    return data


def standard_scale_num_features(
    x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard scales numeric columns in train, validation, and test sets.
    Args:
        x_train (pd.DataFrame): Training features.
        x_val (pd.DataFrame): Validation features.
        x_test (pd.DataFrame): Test features.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Scaled train, val, and test DataFrames.
    """

    x_train_scaled = x_train.copy()
    x_val_scaled = x_val.copy()
    x_test_scaled = x_test.copy()

    ss = StandardScaler()

    x_train_scaled[NUM_COLS] = ss.fit_transform(x_train[NUM_COLS])
    x_test_scaled[NUM_COLS] = ss.transform(x_test[NUM_COLS])
    x_val_scaled[NUM_COLS] = ss.transform(x_val[NUM_COLS])

    return x_train_scaled, x_val_scaled, x_test_scaled


def get_total_users_and_actions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds total actions and total users columns by summing relevant columns.
    Args:
        data (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with total actions and users columns.
    """

    data["TOTAL_ACTIONS"] = (
        data["ACTIONS_CRM_CONTACTS"]
        + data["ACTIONS_CRM_COMPANIES"]
        + data["ACTIONS_CRM_DEALS"]
        + data["ACTIONS_EMAIL"]
    )
    data["TOTAL_USERS"] = (
        data["USERS_CRM_CONTACTS"]
        + data["USERS_CRM_COMPANIES"]
        + data["USERS_CRM_DEALS"]
        + data["USERS_EMAIL"]
    )

    return data


def get_total_cum_actions_and_users(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds total cumulative actions and users columns by summing relevant cumulative columns.
    Args:
        data (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with total cumulative actions and users columns.
    """

    data["TOTAL_CUM_ACTIONS"] = (
        data["CUMULATIVE_ACTIONS_CONTACTS"]
        + data["CUMULATIVE_ACTIONS_COMPANIES"]
        + data["CUMULATIVE_ACTIONS_DEALS"]
        + data["CUMULATIVE_ACTIONS_EMAILS"]
    )
    data["TOTAL_CUM_USERS"] = (
        data["CUMULATIVE_USERS_CONTACTS"]
        + data["CUMULATIVE_USERS_COMPANIES"]
        + data["CUMULATIVE_USERS_DEALS"]
        + data["CUMULATIVE_USERS_EMAILS"]
    )

    return data


def one_hot_encode_categorical(
    data: pd.DataFrame, col: str, modeling: bool = True
) -> pd.DataFrame:
    """
    One-hot encodes a categorical column and drops the original column.
    Args:
        data (pd.DataFrame): Input DataFrame.
        col (str): Name of the categorical column to encode.
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.
    """

    ohe = OneHotEncoder(sparse_output=False, drop="first")
    ohe_df = pd.DataFrame(
        ohe.fit_transform(data[[col]]), columns=ohe.get_feature_names_out([col])
    )
    data = pd.concat(
        [data.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1
    )

    if modeling:
        data = data.drop(columns=[col])

    return data


def label_encode_categorical(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    col: str,
    modeling: bool = True,
) -> pd.DataFrame:
    """
    Label encodes a categorical column and drops the original column.
    Args:
        data (pd.DataFrame): Input DataFrame.
        col (str): Name of the categorical column to encode.
    Returns:
        pd.DataFrame: DataFrame with label encoded column.
    """

    encoder = ce.OrdinalEncoder(handle_unknown="impute")
    X_train[col + "_LE"] = encoder.fit_transform(X_train[col])
    X_val[col + "_LE"] = encoder.transform(X_val[col])
    X_test[col + "_LE"] = encoder.transform(X_test[col])

    if modeling:
        X_train = X_train.drop(columns=[col])
        X_val = X_val.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

    return X_train, X_val, X_test


def cum_actions_growth_comp_to_prev_week(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column for growth in ACTIONS growth compared to the previous week for all usage objects.
    Args:
        data (pd.DataFrame): Input DataFrame with 'id', 'WHEN_TIMESTAMP', and 'ACTIONS_CRM_COMPANIES'.
    Returns:
        pd.DataFrame: DataFrame with 'GROWTH_ACTIONS_COMPANIES_PREV_WEEK' column added.
    """

    data["CONTACTS_CUM_GROWTH"] = data["CUMULATIVE_ACTIONS_CONTACTS"] / data.groupby(
        "id"
    )["CUMULATIVE_ACTIONS_CONTACTS"].shift(1).replace(0, 1)

    data["DEALS_CUM_GROWTH"] = data["CUMULATIVE_ACTIONS_DEALS"] / data.groupby("id")[
        "CUMULATIVE_ACTIONS_DEALS"
    ].shift(1).replace(0, 1)

    data["COMPANIES_CUM_GROWTH"] = data["CUMULATIVE_ACTIONS_COMPANIES"] / data.groupby(
        "id"
    )["CUMULATIVE_ACTIONS_COMPANIES"].shift(1).replace(0, 1)

    data["TOTAL_CUM_GROWTH"] = data["TOTAL_CUM_ACTIONS"] / data.groupby("id")[
        "TOTAL_CUM_ACTIONS"
    ].shift(1).replace(0, 1)

    data["CONTACTS_CUM_GROWTH"] = data["CONTACTS_CUM_GROWTH"].fillna(1)
    data["TOTAL_CUM_GROWTH"] = data["TOTAL_CUM_GROWTH"].fillna(1)
    data["DEALS_CUM_GROWTH"] = data["DEALS_CUM_GROWTH"].fillna(1)
    data["COMPANIES_CUM_GROWTH"] = data["COMPANIES_CUM_GROWTH"].fillna(1)

    return data


def get_features(modeling_data: pd.DataFrame, modeling: bool = False) -> pd.DataFrame:
    """
    Applies a series of feature engineering steps to the modeling data.
    Args:
        modeling_data (pd.DataFrame): Input DataFrame for modeling.
        modeling (bool, optional): If True, applies modeling-specific steps. Defaults to False.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """

    modeling_data = cum_actions_contacts(modeling_data)
    modeling_data = cum_actions_companies(modeling_data)
    modeling_data = cum_actions_deals(modeling_data)
    modeling_data = cum_actions_emails(modeling_data)

    modeling_data = cum_users_contacts(modeling_data)
    modeling_data = cum_users_companies(modeling_data)
    modeling_data = cum_users_deals(modeling_data)
    modeling_data = cum_users_emails(modeling_data)

    modeling_data = get_total_users_and_actions(modeling_data)
    modeling_data = get_total_cum_actions_and_users(modeling_data)
    modeling_data = cum_actions_growth_comp_to_prev_week(modeling_data)

    if modeling:
        modeling_data = rem_outliers(modeling_data)

    modeling_data = fill_na_employee_range(modeling_data)

    modeling_data = log_scale(modeling_data, "ALEXA_RANK", modeling)

    modeling_data = get_date_features(modeling_data, "WHEN_TIMESTAMP", modeling)

    modeling_data = map_employee_range(modeling_data, modeling)

    # modeling_data = one_hot_encode_categorical(modeling_data, "DAY", modeling)

    # if modeling:
    #     modeling_data = modeling_data.drop(["INDUSTRY"], axis=1)  # too many NaNs
    # modeling_data["INDUSTRY"] = modeling_data["INDUSTRY"].fillna("UNKNOWN")
    # modeling_data = one_hot_encode_categorical(modeling_data, "INDUSTRY", modeling)

    return modeling_data.reset_index(drop=True)
