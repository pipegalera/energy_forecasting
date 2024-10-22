from dotenv import load_dotenv
load_dotenv()
import sys
import os
import datetime
from utils import complete_timeframe, create_horizon_dates, create_group_lags, create_group_rolling_means, create_date_colunms
import pandas as pd
import xgboost as xgb
import argparse
import numpy as np
import mlflow
import json

DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
HOME_PATH = os.getenv("HOME_PATH")
MLFLOW_PATH = os.getenv("MLFLOW_PATH")

# Load RUN IDs from MLflow
with open(f"{MLFLOW_PATH}/run_id_mapping.json", "r") as f:
    run_id_mapping = json.load(f)

covs = ['value_lag_3_hours', 'value_lag_6_hours', 'value_lag_12_hours',
       'value_lag_24_hours', 'value_lag_48_hours', 'value_lag_168_hours',
       'value_lag_336_hours', 'value_lag_720_hours', 'value_lag_2160_hours',
       'value_rolling_mean_3_hours', 'value_rolling_mean_6_hours',
       'value_rolling_mean_12_hours', 'value_rolling_mean_24_hours', 'value_rolling_mean_48_hours','value_rolling_mean_168_hours',
       'value_rolling_mean_336_hours',
       'value_rolling_mean_720_hours', 'value_rolling_mean_2160_hours',
       'period_hour', 'period_day', 'period_week', 'period_year',
       'period_day_of_week', 'period_day_of_year', 'period_month_end',
       'period_month_start', 'period_quarter_end', 'period_quarter_start',
       'period_year_end', 'period_year_start']


import mlflow
logged_model = 'runs:/9e3b337f44364bc6a0f4907c432880f5/xgboost_model_VEA'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))



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

    # READ RAW DATA
    df = pd.read_parquet(f"{DATA_PATH}/data.parquet")

    # PREP DATA
    print(f"--> Creating new forecasts...")
    df_horizon = (df
                    .pipe(complete_timeframe, bfill=True)
                    .pipe(create_horizon_dates, 'subba', 14)
                    .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
                    .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
                    .pipe(create_date_colunms, 'period')
         )
    df_horizon = df_horizon.sort_values(['subba', 'period'])


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
