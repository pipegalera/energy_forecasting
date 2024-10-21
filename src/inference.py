from dotenv import load_dotenv
load_dotenv()
import sys
import os
import datetime
from utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums
import pandas as pd
import xgboost as xgb
import argparse
import numpy as np
import mlflow
import json

mlflow.set_tracking_uri(f"sqlite:///mlflow/mlflow.db")
DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")

# Load RUN IDs from MLflow
with open(f"{MODELS_PATH}/run_id_mapping.json", "r") as f:
    run_id_mapping = json.load(f)

covs = ['value_lag_3', 'value_lag_6', 'value_lag_12',
       'value_lag_24', 'value_lag_48', 'value_lag_168', 'value_lag_336',
       'value_lag_720', 'value_lag_2160', 'value_rolling_mean_3_hours',
       'value_rolling_mean_6_hours', 'value_rolling_mean_12_hours',
       'value_rolling_mean_24_hours', 'value_rolling_mean_48_hours',
       'value_rolling_mean_168_hours', 'value_rolling_mean_336_hours',
       'value_rolling_mean_720_hours', 'value_rolling_mean_2160_hours',
       'period_hour', 'period_day', 'period_week', 'period_year',
       'period_day_of_week', 'period_day_of_year', 'period_month_end',
       'period_month_start', 'period_quarter_end', 'period_quarter_start',
       'period_year_end', 'period_year_start']


def create_horizon_dates(data, groups_column, days_before=False, days_after=0):

    df = data.copy()

    horizon = df["period"].max() + datetime.timedelta(days=days_after)
    if days_before:
        last_period = df["period"].max() - datetime.timedelta(days=days_before)
    else:
        last_period = df["period"].max() + datetime.timedelta(hours=1)

    all_predictions_df = pd.DataFrame()
    for i in df[groups_column].unique():
        predictions_df = pd.date_range(last_period, horizon, freq='1h')
        predictions_df = pd.DataFrame({"period":predictions_df})
        predictions_df[groups_column] = i

        all_predictions_df = pd.concat([all_predictions_df, predictions_df]).sort_values([groups_column, 'period'])

    return all_predictions_df

def make_predictions(data, covs = covs):

    df = data.copy()

    predictions = []
    subbas = df["subba"].unique()

    for subba in subbas:

        df_subba = df[df["subba"] == subba][covs]

        # Load model
        run_id = run_id_mapping.get(subba)
        model_uri = f"runs:/{run_id}/xgboost_model_{subba}"
        model = mlflow.xgboost.load_model(model_uri)

        prediction = model.predict(df_subba)
        predictions.extend(prediction)

    df["forecasted_value"] = predictions
    df["forecasted_value"] = df["forecasted_value"].astype('int32')

    df.drop(covs, axis=1, inplace=True)

    return df


def main(days):

    df = pd.read_parquet(f"{DATA_PATH}/data.parquet")

    print(f"--> Creating new forecasts for the next {days} days...")
    df_horizon = create_horizon_dates(data=df,
                                      groups_column='subba',
                                      days_before=days,
                                      days_after=days)

    # PREP DATA
    df_horizon = (df_horizon
                    .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
                    .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
                    .pipe(create_date_colums, 'period')
         )
    df = df.sort_values(['subba', 'period'])

    print("--> Loading models and running predictions...")
    preds = make_predictions(data=df_horizon, covs = covs)

    print("--> Creating new inference file updated...")
    preds = preds.merge(df, on=["period", "subba"], how="left")

    # Limit the data horizon
    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    predictions_cutoff_date = current_time_utc - datetime.timedelta(days=2)
    preds.loc[preds['period'] < predictions_cutoff_date, 'forecasted_value'] = np.nan

    # Delete columns
    preds = preds[["period", "subba", "value", "forecasted_value"]]

    preds.to_parquet(f"{DATA_PATH}/inference.parquet")
    print(f"--> Done! Predictions saved at: {DATA_PATH}/inference.parquet")
    print("---------------------------------------------")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Energy Forecasting Inference')
    parser.add_argument('--days', type=int, default=3, help='Number of days to forecast')
    args = parser.parse_args()
    main(args.days)
