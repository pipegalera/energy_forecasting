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

import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_parquet(DATA_PATH)
df = (df.pipe(complete_timeframe, bfill=True)
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
                 ):


    tss = TimeSeriesSplit(n_splits=n_splits,
                          test_size=24*365*1,
                          gap=24)
    print(f"Training group: {group}...")
    df = data[data["subba"] == group]
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

    print(f"Average RMSE across folds: {np.mean(scores):0.4f}")

    return model

base_model = xgb.XGBRegressor(base_score=0.5,
                         booster='gbtree',
                         n_estimators=1000,
                         early_stopping_rounds=100,
                         objective='reg:squarederror',
                         max_depth=3,
                         learning_rate=0.01,)

xgb_model = ModelTrainer(data=df,
             group="PGAE",
             model=base_model,
             target='value',
             covs=list(df.columns[25:]),
             n_splits=5,
            )
