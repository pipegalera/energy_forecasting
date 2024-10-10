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
from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums, create_horizon

import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_parquet(DATA_PATH)
df = (df.pipe(complete_timeframe, bfill=True)
        .pipe(create_horizon, 'subba', horizon_days=60)
        .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_date_colums, 'period')
     )

def ModelTrainer(data,
                 group,
                 model,
                 target,
                 covs,
                 n_splits=5,
                 save_model=False
                 ):


    tss = TimeSeriesSplit(n_splits=n_splits,
                          test_size=24*365*1,
                          gap=24)
    print(f"Training group: {group}...")
    df = data[(data["subba"] == group) & (data["data_type"] == "Real values")].set_index("period")
    df = df.sort_index()

    preds = []
    scores = []
    xgb_model = None
    for i, (train_idx, test_idx) in enumerate(tss.split(df)):
        print(f"Training loop {i}...")
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        X_train = train[covs]
        y_train = train[target]
        X_test = test[covs]
        y_test = test[target]

        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100,
                xgb_model=xgb_model)

        xgb_model = model.get_booster() # Enable incremental learning

        y_pred = model.predict(X_test)
        score = root_mean_squared_error(y_test,y_pred)

        preds.append(y_pred)
        scores.append(score)

    if save_model:
        xgb_model.save_model(f"models/xgb/xgb_model_{group}.json")

    print(f"Average RMSE across folds: {np.mean(scores):0.4f}")

    return model


base_model = xgb.XGBRegressor(base_score=0.5,
                         booster='gbtree',
                         n_estimators=1000,
                         early_stopping_rounds=100,
                         objective='reg:squarederror',
                         max_depth=3,
                         learning_rate=0.01,)

covs = ['period_day',
        'period_week', 'period_year',
        'period_day_of_week','period_day_of_year',
        'period_month_end', 'period_month_start',
        'period_year_end', 'period_year_start',
        'period_quarter_end', 'period_quarter_start',]

for subba in df["subba"].unique():
    xgb_model = ModelTrainer(data=df,
                group=subba,
                model=base_model,
                target='value',
                covs=covs,
                n_splits=5,
                save_model=True
                )
