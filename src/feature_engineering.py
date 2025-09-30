import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def cum_actions_contacts(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_ACTIONS_CONTACTS'] = data.groupby('id')['ACTIONS_CRM_CONTACTS'].cumsum()

    return data

def cum_actions_companies(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_ACTIONS_COMPANIES'] = data.groupby('id')['ACTIONS_CRM_COMPANIES'].cumsum()

    return data

def cum_actions_deals(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_ACTIONS_DEALS'] = data.groupby('id')['ACTIONS_CRM_DEALS'].cumsum()

    return data

def cum_actions_emails(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_ACTIONS_EMAILS'] = data.groupby('id')['ACTIONS_EMAIL'].cumsum()

    return data


def cum_users_contacts(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_USERS_CONTACTS'] = data.groupby('id')['USERS_CRM_CONTACTS'].cumsum()

    return data

def cum_users_companies(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_USERS_COMPANIES'] = data.groupby('id')['USERS_CRM_COMPANIES'].cumsum()

    return data

def cum_users_deals(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_USERS_DEALS'] = data.groupby('id')['USERS_CRM_DEALS'].cumsum()

    return data

def cum_users_emails(data):

    data = data.sort_values(by=['id', 'WHEN_TIMESTAMP'])
    data['CUMULATIVE_USERS_EMAILS'] = data.groupby('id')['USERS_EMAIL'].cumsum()

    return data

def get_date_features(data, col, modeling=True):

    data[col] = pd.to_datetime(data[col])
    data['DAY'] = data[col].dt.day
    data['MONTH'] = data[col].dt.month
    data['YEAR'] = data[col].dt.year
    if modeling:
        data = data.drop(["WHEN_TIMESTAMP"], axis=1)

    return data

def rem_outliers(data):

    data = data[data["ACTIONS_CRM_CONTACTS"] < 3000]
    data = data[data["ACTIONS_CRM_COMPANIES"] < 1000]
    data = data[data["ACTIONS_CRM_DEALS"] < 1000]
    data = data[data["USERS_CRM_COMPANIES"] < 30]
    data = data[data["USERS_CRM_DEALS"] < 40]
    data = data[data["ACTIONS_EMAIL"] < 175]
    
    return data

def fill_na_employee_range(data):

    if data[data['EMPLOYEE_RANGE'].isna()]["ALEXA_RANK"].values[0] > 15000001.0:

        data['EMPLOYEE_RANGE'] = data['EMPLOYEE_RANGE'].fillna("1")

    return data

def log_scale(data, col, modeling=True):

    data['ALEXA_RANK_CLEAN'] = data['ALEXA_RANK'].replace(0, np.nan)
    data['ALEXA_RANK_LOG'] = np.log1p(data['ALEXA_RANK_CLEAN'])

    if modeling:
        data = data.drop(["ALEXA_RANK", "ALEXA_RANK_CLEAN"], axis=1)
    
    return data

def cyclic_encode_day(data, modeling=True):

    data['day_sin'] = np.sin(2 * np.pi * data["DAY"]/31)

    if modeling:
        data = data.drop(['DAY'], axis=1)

def cyclic_encode_month(data, modeling=True):

    data['month_sin'] = np.sin(2 * np.pi * data["MONTH"]/31)

    if modeling:
        data = data.drop(['MONTH'], axis=1)

def map_employee_range(data, modeling=True):

    employee_range_map = {"1":1, "2 to 5":2, "6 to 10":3, "11 to 25":4, "26 to 50":5, "51 to 200":6, "201 to 1000":7, "1001 to 10000":8, "10,001 or more":9}

    data['EMPLOYEE_RANGE_MAPPED'] = data['EMPLOYEE_RANGE'].map(employee_range_map)

    if modeling:
        data = data.drop(["EMPLOYEE_RANGE"], axis=1)
    
    return data

def standard_scale_num_features(x_train, x_val, x_test):

    NUM_COLS = ["ACTIONS_CRM_CONTACTS", "ACTIONS_CRM_COMPANIES", "ACTIONS_CRM_DEALS", "ACTIONS_EMAIL", "USERS_CRM_CONTACTS", "USERS_CRM_COMPANIES",
            "USERS_CRM_DEALS", "USERS_EMAIL", "ALEXA_RANK_LOG", "CUMULATIVE_ACTIONS_CONTACTS", "CUMULATIVE_ACTIONS_COMPANIES", "CUMULATIVE_ACTIONS_DEALS", 
            "CUMULATIVE_ACTIONS_EMAILS", "CUMULATIVE_USERS_CONTACTS", "CUMULATIVE_USERS_COMPANIES", "CUMULATIVE_USERS_DEALS", "CUMULATIVE_USERS_EMAILS",
            "YEAR"]

    x_train_scaled = x_train.copy()
    x_val_scaled = x_val.copy()
    x_test_scaled = x_test.copy()

    ss = StandardScaler()

    x_train_scaled[NUM_COLS] = ss.fit_transform(x_train[NUM_COLS])
    x_test_scaled[NUM_COLS] = ss.transform(x_test[NUM_COLS])
    x_val_scaled[NUM_COLS] = ss.transform(x_val[NUM_COLS])

    return x_train_scaled, x_val_scaled, x_test_scaled


def get_features(modeling_data, modeling=False):

    modeling_data = cum_actions_contacts(modeling_data)
    modeling_data = cum_actions_companies(modeling_data)
    modeling_data = cum_actions_deals(modeling_data)
    modeling_data = cum_actions_emails(modeling_data)

    modeling_data = cum_users_contacts(modeling_data)
    modeling_data = cum_users_companies(modeling_data)
    modeling_data = cum_users_deals(modeling_data)
    modeling_data = cum_users_emails(modeling_data)

    if modeling:
        modeling_data = rem_outliers(modeling_data)

    modeling_data = fill_na_employee_range(modeling_data)

    modeling_data = log_scale(modeling_data, "ALEXA_RANK", modeling)

    modeling_data = get_date_features(modeling_data, "WHEN_TIMESTAMP", modeling)

    modeling_data = map_employee_range(modeling_data, modeling)

    # modeling_data = cyclic_encode_day(modeling_data, modeling)
    # modeling_data = cyclic_encode_month(modeling_data, modeling)

    if modeling:
        modeling_data = modeling_data.drop(["INDUSTRY"], axis=1) # too many NaNs

    return modeling_data.reset_index(drop=True)