from dotenv import load_dotenv
load_dotenv()
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.xgboost
import optuna
import json

HOME_PATH = os.getenv("HOME_PATH")
DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")

os.chdir(HOME_PATH)
sys.path.append(os.getcwd())

from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_columns



# SET EXPERIMENT
# bash -> mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
experiment_name = "XGBoost_Energy_Forecasting"
mlflow.set_experiment(experiment_name)


# READ DATA
df = pd.read_parquet(DATA_PATH + "data.parquet")

# PREP DATA
df = (df
        .pipe(complete_timeframe, bfill=True)
        .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_date_columns, 'period')
     )
df = df.sort_values(['subba', 'period'])


# Covariates
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

# Optuna XGBoost model
def objective(trial, data, group, target, covs, n_splits=5):
    params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
        }

    model = xgb.XGBRegressor(early_stopping_rounds=100, **params)

    tss = TimeSeriesSplit(n_splits=n_splits, test_size=24*365*1, gap=24)
    df_group = data[(data["subba"] == group)].set_index("period").sort_index()

    rmse_scores = []
    mape_scores = []
    for i, (train_idx, test_idx) in enumerate(tss.split(df_group)):
        train = df_group.iloc[train_idx]
        test = df_group.iloc[test_idx]
        X_train, y_train = train[covs], train[target]
        X_test, y_test = test[covs], test[target]

        model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
                )

        y_pred = model.predict(X_test)

        rmse_score = root_mean_squared_error(y_test, y_pred)
        mape_score = mean_absolute_percentage_error(y_test, y_pred)
        rmse_scores.append(rmse_score)
        mape_scores.append(mape_score)

    return np.mean(rmse_scores), np.mean(mape_scores)

def optimize_and_train(data, group, target, covs, n_splits=5, n_trials=100):
    with mlflow.start_run(run_name=f"XGBoost_Optuna_{group}",nested=True) as run:
        mlflow.log_param("group", group)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(directions=['minimize', 'minimize'])
        study.optimize(lambda trial: objective(trial, data, group, target, covs, n_splits), n_trials=n_trials)

        best_params = study.best_trials[-1].params
        best_rmse, best_mape = study.best_trials[-1].values

        mlflow.log_params(best_params)
        mlflow.log_metric("best_RMSE", best_rmse)
        mlflow.log_metric("best_MAPE", best_mape)

        # Train final model with best params
        best_params['objective'] = 'reg:squarederror'
        best_params['eval_metric'] = 'rmse'
        best_model = xgb.XGBRegressor(**best_params)

        df_group = data[(data["subba"] == group)].set_index("period").sort_index()
        X, y = df_group[covs], df_group[target]
        best_model.fit(X, y)

        mlflow.xgboost.log_model(best_model,
                                 f"xgboost_model_{group}",
                                 input_example=X.iloc[:5])

        # Calculate and log MAPE for the entire dataset
        y_pred = best_model.predict(X)
        final_mape = mean_absolute_percentage_error(y, y_pred)
        mlflow.log_metric("final_MAPE", final_mape)

        # Save run id
        run_id = run.info.run_id
        mapping_file = f"{MODELS_PATH}/run_id_mapping.json"

        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                run_id_mapping = json.load(f)
        else:
            run_id_mapping = {}

        run_id_mapping[group] = run_id

        with open(mapping_file, 'w') as f:
            json.dump(run_id_mapping, f)

    mlflow.end_run()

    return best_model

# Optimize and train the model for each group
subbas = df["subba"].unique()
for subba in subbas:
    best_model = optimize_and_train(
        data=df,
        group=subba,
        target='value',
        covs=covs,
        n_splits=5,
        n_trials=100
    )
    print(f"Finished optimization and training for group: {subba}")
