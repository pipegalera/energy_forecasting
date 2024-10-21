from dotenv import load_dotenv
load_dotenv()
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
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

from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums

# SET EXPERIMENT
mlflow_dir = "mlflow"
if not os.path.exists(mlflow_dir):
    os.makedirs(mlflow_dir)
mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir}/mlflow.db")
mlflow.set_experiment("XGBoost_Energy_Forecasting")
print("MLflow tracking URI:", mlflow.get_tracking_uri())
# mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

# READ DATA
df = pd.read_parquet(DATA_PATH + "data.parquet")

# PREP DATA
df = (df.pipe(complete_timeframe, bfill=True)
        .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_date_colums, 'period')
     )
df = df.sort_values(['subba', 'period'])


# Covariates
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

    scores = []
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
        score = root_mean_squared_error(y_test, y_pred)
        scores.append(score)

    return np.mean(scores)

def optimize_and_train(data, group, target, covs, n_splits=5, n_trials=100):
    with mlflow.start_run(run_name=f"XGBoost_Optuna_{group}") as run:
        mlflow.log_param("group", group)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, data, group, target, covs, n_splits), n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_RMSE", best_value)

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
