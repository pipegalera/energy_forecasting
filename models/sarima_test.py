from dotenv import load_dotenv
load_dotenv()
import os
HOME_PATH = os.getenv("HOME_PATH")
DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")

os.chdir(HOME_PATH)
import pandas as pd
import numpy as np
import datetime
from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums


df = pd.read_parquet(DATA_PATH)
df = (df.pipe(complete_timeframe, bfill=True)
        .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_date_colums, 'period')
     )

def train_sarima(data, target='value', covs=None, save_model=True):

    models = {}
    subbas = df['subba'].unique()

    for subba in subbas:
        # Target variable
        y = data[data['subba'] == subba][target]

        # Covariates
        # covs = data[['value_lag_3', 'value_lag_6', 'value_lag_12',
        #         'value_lag_24', 'value_lag_48', 'value_lag_168', 'value_lag_336',
        #         'value_lag_720', 'value_lag_2160', 'value_rolling_mean_3_hours',
        #         'value_rolling_mean_6_hours', 'value_rolling_mean_12_hours',
        #         'value_rolling_mean_24_hours', 'value_rolling_mean_48_hours',
        #         'value_rolling_mean_168_hours', 'value_rolling_mean_336_hours',
        #         'value_rolling_mean_720_hours', 'value_rolling_mean_2160_hours',
        #         'period_day', 'period_week', 'period_year', 'period_day_of_week',
        #         'period_day_of_year', 'period_month_end', 'period_month_start',
        #         'period_quarter_end', 'period_quarter_start', 'period_year_end',
        #         'period_year_start']]

        print(f"Training SARIMA for {subba} region")
        model = SARIMAX(y,
                        #exog=covs, # Lags contain NaN because data do not exist
                        order=(1,1,1),
                        seasonal_order=(1,1,1,24))
        results = model.fit(method='lbfgs', maxiter=5000, disp=True)
        models[subba] = model

        # Save the model
        if save_model:
            model_filename = os.path.join(MODELS_PATH, f'sarima_model_{subba}.pkl')
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)
            print(f"Model for {subba} saved to {model_filename}")
