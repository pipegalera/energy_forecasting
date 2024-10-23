from dotenv import load_dotenv
load_dotenv()
import sys
import os
import datetime
from utils import complete_timeframe, create_horizon_dates, create_group_lags, create_group_rolling_means, create_date_columns
import pandas as pd
import xgboost as xgb
import argparse
import numpy as np
import mlflow
import json

DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
HOME_PATH = os.getenv("HOME_PATH")


# Load RUN IDs from MLflow
with open(f"{MODELS_PATH}/run_id_mapping.json", "r") as f:
    run_id_mapping = json.load(f)
experiment_id = "351954414414469151"


def make_predictions(data):
    df = data.copy()

    predictions = []
    subbas = df["subba"].unique()

    for subba in subbas:
        df_subba = df[df["subba"] == subba].copy()
        df_subba = df_subba.drop(['period', 'subba', 'subba-name', 'parent', 'parent-name', 'value', 'value-units'],
            axis=1)

        # Load model
        model_uri = f"{HOME_PATH}/mlartifacts/{experiment_id}/{run_id_mapping[subba]}/artifacts/xgboost_model_{subba}"
        model = mlflow.xgboost.load_model(model_uri)

        prediction = model.predict(df_subba)
        predictions.extend(prediction)

    df["forecasted_value"] = predictions
    df["forecasted_value"] = df["forecasted_value"].astype('int32')

    return df


def main(days=14):

    # READ RAW DATA
    df = pd.read_parquet(f"{DATA_PATH}/data.parquet")

    # PREP DATA
    print(f"--> Creating new forecasts...")
    df_horizon = (df
                    .pipe(complete_timeframe, bfill=True)
                    .pipe(create_horizon_dates, 'subba', 14)
                    .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
                    .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
                    .pipe(create_date_columns, 'period')
         )
    df_horizon = df_horizon.sort_values(['subba', 'period'])

    df_horizon.period.max()

    # Limit the data
    cutoff_date = pd.Timestamp.utcnow()  - datetime.timedelta(days=days)
    df_horizon = df_horizon.loc[df_horizon['period'] > cutoff_date]
    print("--> Min date:", df_horizon.period.min())
    print("--> Max date:", df_horizon.period.max())

    print("--> Loading models and running predictions...")
    preds = make_predictions(data=df_horizon)

    # Limit the horizon shows
    df_horizon.loc[df_horizon['period'] < cutoff_date, 'forecasted_value'] = np.nan


    # Delete columns
    preds = preds[["period", "subba", "value", "forecasted_value"]]
    preds

    preds.to_parquet(f"{DATA_PATH}/inference.parquet")
    print(f"--> Done! Predictions saved at: {DATA_PATH}/inference.parquet")
    print("---------------------------------------------")

if __name__=="__main__":
    main()
