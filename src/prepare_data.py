import pandas as pd
from src.config import FileArgs
from sklearn.model_selection import train_test_split
from src.feature_engineering import get_features, standard_scale_num_features
import logging

logging.basicConfig(
    level=logging.INFO,  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


def prepare_weekly_usage_data_for_everyone(fileargs: FileArgs, output_file_path=None):
    """Combines given files to create the final weekly report data"""

    customers_df = pd.read_csv(fileargs.customers_path)
    noncustomers_df = pd.read_csv(fileargs.non_customers_path)
    usage_actions_df = pd.read_csv(fileargs.usage_path)

    # create a flag for train and test set
    # customers data will be used for training
    customers_df['customers'] = "yes"
    noncustomers_df['customers'] = "no"

    metadata = pd.concat([customers_df, noncustomers_df], ignore_index=True)

    # Duplicates rows present for IDs 280, 278, 279
    metadata.drop_duplicates(subset=["id"], inplace=True)

    complete_report_data = pd.merge(usage_actions_df, metadata, how='left', on='id').sort_values(by="WHEN_TIMESTAMP")

    complete_report_data['WHEN_TIMESTAMP'] = pd.to_datetime(complete_report_data['WHEN_TIMESTAMP'])

    if output_file_path:
        complete_report_data.to_csv(output_file_path, index=False)
    
    return complete_report_data


def prepare_modeling_data(fileargs: FileArgs, feature_engineer = True, output_file_path=None, modeling=True):
    """set modeling = True for modeling data and False to retrieve EDA data"""
    customers_df = pd.read_csv(fileargs.customers_path)
    usage_actions_df = pd.read_csv(fileargs.usage_path)

    customers_df['CLOSEDATE'] = pd.to_datetime(customers_df['CLOSEDATE'])
    usage_actions_df['WHEN_TIMESTAMP'] = pd.to_datetime(usage_actions_df['WHEN_TIMESTAMP'])

    customers_df.drop_duplicates(subset=["id"], inplace=True) # drop any duplicates if present

    modeling_df = pd.merge(usage_actions_df, customers_df[['id','CLOSEDATE','ALEXA_RANK','EMPLOYEE_RANGE','INDUSTRY']], 
                    how='left', on='id')

    modeling_df['DAYS_DIFF'] = (modeling_df['CLOSEDATE'] - modeling_df['WHEN_TIMESTAMP']).dt.days

    modeling_df = modeling_df[modeling_df['DAYS_DIFF'] > 0] # select rows before conversion date

    modeling_df['WILL_CONVERT'] = (modeling_df['DAYS_DIFF'] <= 30).astype(int)

    if feature_engineer:
        modeling_df = get_features(modeling_df, modeling)

    if modeling:
        modeling_df.drop(["CLOSEDATE", "DAYS_DIFF"], axis=1, inplace=True)

    if output_file_path:
        modeling_df.to_csv(output_file_path, index=False)

    return modeling_df


def get_train_test_val_split(modeling_data, scale_num_features= True, train_perc = 0.7):

    # Extract unique IDs
    unique_ids = modeling_data['id'].unique()

    train_ids, temp_ids = train_test_split(unique_ids, test_size=1-train_perc, random_state=42)

    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=23)

    # Create datasets based on IDs
    train_df = modeling_data[modeling_data['id'].isin(train_ids)].copy()
    val_df = modeling_data[modeling_data['id'].isin(val_ids)].copy()
    test_df = modeling_data[modeling_data['id'].isin(test_ids)].copy()

    X_train = train_df.drop(columns=['WILL_CONVERT', "id"])
    y_train = train_df['WILL_CONVERT']

    X_val = val_df.drop(columns=['WILL_CONVERT', "id"])
    y_val = val_df['WILL_CONVERT']

    X_test = test_df.drop(columns=['WILL_CONVERT', "id"])
    y_test = test_df['WILL_CONVERT']

    logger.info(f"Train:, {X_train.shape, y_train.shape}")
    logger.info(f"Validation:, {X_val.shape, y_val.shape}")
    logger.info(f"Test:, {X_test.shape, y_test.shape}")

    if scale_num_features:
        x_train_scaled, x_val_scaled, x_test_scaled = standard_scale_num_features(X_train, X_val, X_test)
    
    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test
 