from dotenv import load_dotenv
load_dotenv()
import sys
import os
import datetime
from utils import create_date_colums
import pandas as pd
import xgboost as xgb

DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")

covariates = ['period_hour','period_day','period_week', 'period_year',
        'period_day_of_week','period_day_of_year',
        'period_month_end', 'period_month_start',
        'period_year_end', 'period_year_start',
        'period_quarter_end', 'period_quarter_start',]


def create_horizon_dates(data, groups_column, days_after, days_before=False):

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

def make_predictions(data, covs = covariates, save_file=True):

    df = data.copy()

    model = xgb.XGBRegressor()
    predictions = []
    subbas = df["subba"].unique()

    for subba in subbas:
        # Data
        df_subba = df[df["subba"]== subba][covs]

        # Model
        model.load_model(f"{MODELS_PATH}/xgb/xgb_model_{subba}.json")
        prediction = model.predict(df_subba)
        predictions.extend(prediction)

    df["forecasted_value"] = predictions

    # Formatting
    df["forecasted_value"] = df["forecasted_value"].astype('int32')

    df.drop(covs, axis=1, inplace=True)

    return df


if __name__=="__main__":

    df = pd.read_parquet(f"{DATA_PATH}/data.parquet")
    days = 30

    print(f"--> Creating new forecasts for the next {days} days")
    df_horizon = create_horizon_dates(data=df,
                                    groups_column='subba',
                                    days_after=days,
                                    days_before=days)
    df_horizon = create_date_colums(df_horizon, 'period')

    print(f"--> Loading models and running predictions...")
    df_inference = make_predictions(data=df_horizon, covs = covariates, save_file=False)

    output_file = f"{DATA_PATH}/inference.parquet"
    if os.path.exists(output_file):
            print(f"--> Appending new data to inference.parquet...")
            existing_df = pd.read_parquet(output_file)
            df_inference = df.merge(df_inference, on=["period", "subba"], how="outer")
            df_inference = pd.concat([existing_df, df_inference]).drop_duplicates(subset=["period", "subba"])
    else:
        print(f"--> Did not find inference.parquet. Creating new file...")
        df_inference = df.merge(df_inference, on=["period", "subba"], how="outer")

    df_inference.to_parquet(f"{DATA_PATH}/inference.parquet")
    print(f"--> Done! Predictions saved at: {DATA_PATH}/inference.parquet")
    print("---------------------------------------------")
    print(df_inference.head())
