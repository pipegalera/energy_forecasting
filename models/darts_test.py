import pandas as pd
import numpy as np

from darts import TimeSeries
from darts.models import ExponentialSmoothing, ARIMA, VARIMA, NBEATSModel
from darts.metrics import mape, mae, rmse, smape
from darts.utils.utils import ModelMode
from darts.models import NaiveDrift
from darts import concatenate
from darts.utils.callbacks import TFMProgressBar
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


DATA_PATH = "/Users/pipegalera/dev/Documents/GitHub/energy_forecasting/data/data.parquet"
df = pd.read_parquet(DATA_PATH)

df.period = df.period.dt.tz_localize(None)
df = df[["period", "subba", "value"]]
subba_names = df['subba'].unique()


def generate_torch_kwargs():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return {
        "pl_trainer_kwargs": {
            "accelerator": device,
            "devices": 1,
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
            "precision": '32-true',  # Force 32-bit precision
        },
         "force_reset": True
        }


def print_testing(forecasts, train, test):
    print("Type of forecasts:", type(forecasts))
    print("Number of components in forecasts:", forecasts.n_components)
    print("Width of forecasts:", forecasts.width)
    print("Start time of forecasts:", forecasts.start_time())
    print("End time of forecasts:", forecasts.end_time())

    print("\nType of test:", type(test))
    print("Length of test:", len(test))

    if len(test) > 0:
        print("\nType of first test item:", type(test[0]))
        print("Test size:", len(test[0]))
        print("Number of components in first test item:", test[0].n_components)
        print("Width of first test item:", test[0].width)
        print("Start time of first test item:", test[0].start_time())
        print("End time of first test item:", test[0].end_time())

    print("\nType of train:", type(train))
    print("Length of train:", len(train))

    if len(train) > 0:
        print("\nType of first train item:", type(train[0]))
        print("Train size:", len(train[0]))
        print("Number of components in first train item:", train[0].n_components)
        print("Width of first train item:", train[0].width)
        print("Start time of first train item:", train[0].start_time())
        print("End time of first train item:", train[0].end_time())

def complete_timeframe(df):

    date_range = pd.date_range(start=df.period.min(),
                        end=df.period.max(),
                        freq='h')
    df_combined = pd.DataFrame()
    for subba in df.subba.unique():
        df_subba = pd.DataFrame({"period": date_range, "subba": subba})
        df_combined = pd.concat([df_combined, df_subba])

    if len(df_combined) == len(date_range)*4:
        df = df_combined.merge(df, on=['period', 'subba'], how='left')

    df['value'] = df.groupby('subba')['value'].bfill()

    return df

def timeseries_train_test_split(series, test_size=0.8):
    train = []
    test = []
    for ts in series:
        train_part, test_part = ts.split_before(test_size)
        group_name = ts.static_covariates['subba'].iloc[0]

        # Ensure we're working with 1D data
        train_values = train_part.values().ravel()
        test_values = test_part.values().ravel()

        train_df = pd.DataFrame(
            {group_name: train_values},
            index=train_part.time_index)
        test_df = pd.DataFrame(
            {group_name: test_values},
            index=test_part.time_index)

        train_ts = TimeSeries.from_dataframe(train_df)
        test_ts = TimeSeries.from_dataframe(test_df)

        train.append(train_ts)
        test.append(test_ts)

    return train, test

def forecasting(train, test, model):
    combined_train = concatenate(train, axis=1)
    model.fit(combined_train)
    horizon = len(test[0])
    forecasts = model.predict(horizon)
    return forecasts

def evaluate(forecasts, test):
    for i, x in enumerate(forecasts.components):
        print(f"Evaluating component: {x}")
        print("Using metric: RMSE")
        metric = rmse(test[i], forecasts[x])
        print(f"RMSE: {metric:.2f}")
        print()

def timeseries_scatterplot(train, test, forecasts):
    fig = go.Figure()
    dropdown_options = []
    component_mapping = dict(zip(forecasts.components, subba_names))

    for i, component in enumerate(forecasts.components):
        subba_name = component_mapping[component]
        # Train set
        train_values = (train[i].pd_dataframe()
                        .unstack()
                        .reset_index()
                        .set_index("period")
                        .rename({0: "value"}, axis=1)
                        .drop("component", axis=1)
                        .asfreq('h'))

        # Test set
        test_values = (test[i].pd_dataframe()
                    .unstack()
                    .reset_index()
                    .set_index("period")
                    .rename({0: "value"}, axis=1)
                    .drop("component", axis=1)
                    .asfreq('h'))
        test_values_ts = TimeSeries.from_dataframe(test_values)

        # Forecast
        component_forecast = forecasts[component]

        # Add traces for each component
        fig.add_trace(
            go.Scatter(x=train_values.index, y=train_values["value"], name=f"{subba_name} - Train Values", mode='lines', visible=False)
        )
        fig.add_trace(
            go.Scatter(x=test_values.index, y=test_values["value"], name=f"{subba_name} - Test Values", mode='lines', visible=False)
        )
        fig.add_trace(
            go.Scatter(x=component_forecast.pd_dataframe().index, y=component_forecast.pd_dataframe()[f"{component}"],
                    name=f"{subba_name} - Forecast", mode='lines', visible=False)
        )

        # Create dropdown menu option for this component
        dropdown_options.append(
            dict(
                method='update',
                label=subba_name,
                args=[{'visible': [True if j // 3 == i else False for j in range(len(forecasts.components) * 3)]}]
            )
        )

    # Set the first component as initially visible
    for i in range(3):
        fig.data[i].visible = True

    # Update layout
    fig.update_layout(
        title_text="California Electricity Demand",
        xaxis_title="Date",
        yaxis_title="Demand (Mhw)",
        template="plotly_white",
        height=600,
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_options,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.9,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ],
    annotations=[
        dict(
                        text="Subregion:",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.82,
                        y=1.09,
                        xanchor="right",
                        yanchor="middle",
                        font=dict(size=14),
                    ),
        dict(
            text="* Hourly demand by Californian Independent System Operatore. Source: Form EIA-930 Product: Hourly Electric Grid Monitor (https://www.eia.gov/opendata/)",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.15,  # Adjust this value to position the text vertically
            xanchor="left",
            yanchor="top",
            font=dict(size=10),
            )
        ]
    )
    # Show the plot
    return pio.write_html(fig, file='plot.html', auto_open=True)

if __name__=="__main__":
    df = complete_timeframe(df)

    torch.manual_seed(1)
    np.random.seed(1)
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=100,
        random_state=0,
        **generate_torch_kwargs()
    )
    #model = NaiveDrift()

    series = TimeSeries.from_group_dataframe(
                                df=df,
                                group_cols='subba',
                                time_col='period',
                                value_cols='value',
                                freq='h')
    train, test = timeseries_train_test_split(series, test_size=0.8)
    forecasts = forecasting(train, test, model)
    #evaluate(forecasts, test)
    timeseries_scatterplot(train, test, forecasts)
