import pandas as pd
import numpy as np
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

import pickle

DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
df = pd.read_parquet(DATA_PATH)


def complete_timeframe(data, bfill=False):

    df = data.copy()

    df["period"] = df["period"].dt.tz_localize(None)
    date_range = pd.date_range(start=df.period.min(),
                        end=df.period.max(),
                        freq='h')
    df_combined = pd.DataFrame()
    for subba in df.subba.unique():
        df_subba = pd.DataFrame({"period": date_range, "subba": subba})
        df_combined = pd.concat([df_combined, df_subba])

    if len(df_combined) == len(date_range)*4:
        df = df_combined.merge(df, on=['period', 'subba'], how='left')
    if bfill==True:
        df['value'] = df.groupby('subba')['value'].bfill()

    return df
def create_group_lags(data, group_column, target_columns, lags):

    df = data.copy()
    for col in target_columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby(group_column)[col].shift(lag)
        return df
def create_group_rolling_means(data, group_column, target_columns, windows):

    df = data.copy()

    df = df.sort_values(by=[group_column]).reset_index(drop=True)
    for col in target_columns:
        for window in windows:
            new_col_name = f'{col}_rolling_mean_{window}_hours'
            df[new_col_name] = (
                df.groupby(group_column)[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    return df
def create_date_colums(data, date_column):

    df = data.copy()

    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_week'] = df[date_column].dt.isocalendar().week
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_day_of_week'] = df[date_column].dt.dayofweek
    df[f'{date_column}_day_of_year'] = df[date_column].dt.dayofyear
    df[f'{date_column}_month_end'] = df[date_column].dt.is_month_end
    df[f'{date_column}_month_start'] = df[date_column].dt.is_month_start
    df[f'{date_column}_quarter_end'] = df[date_column].dt.is_quarter_end
    df[f'{date_column}_quarter_start'] = df[date_column].dt.is_quarter_start
    df[f'{date_column}_year_end'] = df[date_column].dt.is_quarter_end
    df[f'{date_column}_year_start'] = df[date_column].dt.is_quarter_start

    return df


df = (df.pipe(complete_timeframe, bfill=True)
        .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_date_colums, 'period')
     )


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    if result[1] <= 0.01:
        return  print("Augmented Dickey-Fuller unit root test shows clear stationality")
    else:
        return print("Stationality cannot be proven at 1%")


models = {}
groups = df['subba'].unique()

for subba in groups:
    print(f"Training SARIMA for {subba} region")

    # Target variable
    y = df[df['subba'] == subba]['value']

    # Covariates
    covs = df[['value_lag_3', 'value_lag_6', 'value_lag_12',
            'value_lag_24', 'value_lag_48', 'value_lag_168', 'value_lag_336',
            'value_lag_720', 'value_lag_2160', 'value_rolling_mean_3_hours',
            'value_rolling_mean_6_hours', 'value_rolling_mean_12_hours',
            'value_rolling_mean_24_hours', 'value_rolling_mean_48_hours',
            'value_rolling_mean_168_hours', 'value_rolling_mean_336_hours',
            'value_rolling_mean_720_hours', 'value_rolling_mean_2160_hours',
            'period_day', 'period_week', 'period_year', 'period_day_of_week',
            'period_day_of_year', 'period_month_end', 'period_month_start',
            'period_quarter_end', 'period_quarter_start', 'period_year_end',
            'period_year_start']]


    model = SARIMAX(y,
                    #exog=covs, # Lags contain NaN because data do not exist
                    order=(1,1,1),
                    seasonal_order=(1,1,1,24))
    results = model.fit(method='lbfgs', maxiter=5000, disp=True)
    models[subba] = model

    # Save the model
    model_filename = os.path.join(MODELS_PATH, f'sarima_model_{subba}.pkl')
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model for {subba} saved to {model_filename}")
