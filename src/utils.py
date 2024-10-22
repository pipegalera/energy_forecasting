import pandas as pd
import datetime
#from statsmodels.tsa.stattools import adfuller

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

def create_horizon_dates(data, groups_column, days_after=0):

    df = data.copy()

    last_period = df["period"].max() + datetime.timedelta(hours=1)
    horizon = df["period"].max() + datetime.timedelta(days=days_after)

    all_predictions_df = pd.DataFrame()
    for i in df[groups_column].unique():
        predictions_df = pd.date_range(last_period, horizon, freq='1h')
        predictions_df = pd.DataFrame({"period":predictions_df})
        predictions_df[groups_column] = i

        all_predictions_df = pd.concat([all_predictions_df, predictions_df])

    return pd.concat([df, all_predictions_df]).sort_values([groups_column, 'period'])


def create_group_lags(data, group_column, target_columns, lags):
    df = data.copy()
    for col in target_columns:
        for lag in lags:
            df[f'{col}_lag_{lag}_hours'] = df.groupby(group_column)[col].shift(lag)
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

def create_date_columns(data, date_column):

    df = data.copy()

    df[f'{date_column}_hour'] = df[date_column].dt.hour
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_week'] = df[date_column].dt.isocalendar().week.astype('int32')
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





# def test_stationarity(timeseries):
#     result = adfuller(timeseries, autolag='AIC')
#     if result[1] <= 0.01:
#         return  print("Augmented Dickey-Fuller unit root test shows clear stationality")
#     else:
#         return print("Stationality cannot be proven at 1%")
