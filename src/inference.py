from dotenv import load_dotenv
load_dotenv()
import sys
import os
DATA_PATH = os.getenv("DATA_PATH")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import create_date_colums, create_horizon
import pandas as pd
import xgboost as xgb

HOME_PATH = os.getenv("HOME_PATH")
DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")

covariates = ['period_hour','period_day','period_week', 'period_year',
        'period_day_of_week','period_day_of_year',
        'period_month_end', 'period_month_start',
        'period_year_end', 'period_year_start',
        'period_quarter_end', 'period_quarter_start',]

def make_predictions(data, covs = covariates, save_file=True):

    df = data.copy()

    model = xgb.XGBRegressor()
    predictions = []
    subbas = df["subba"].unique()

    for subba in subbas:
        # Data
        df_subba = df[df["subba"]== subba][covs]

        # Model
        model.load_model(MODELS_PATH + f"/xgb/xgb_model_{subba}.json")
        prediction = model.predict(df_subba)
        predictions.extend(prediction)

    # Insert in horizon df
    df["value"] = predictions

    if save_file:
        df.to_parquet(DATA_PATH + "inference.parquet", index=False)
    else:
        return df

if __name__=="__main__":
    df = pd.read_parquet(DATA_PATH + "data_preprocessed.parquet")
    horizon_days = 30
    print(f"--> Creating new predictions for the next {horizon_days} days inference.parquet...")

    # Create df of future dates
    horizon = create_horizon(data=df,
                            groups_column='subba',
                            horizon_days=horizon_days)

    # Add date features
    horizon = create_date_colums(df, 'period')

    # Inference
    make_predictions(horizon)
    print("--> Done! Predictions saved at:", DATA_PATH)
    print("---------------------------------------------")
