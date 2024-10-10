from dotenv import load_dotenv
load_dotenv()
import os
import sys


HOME_PATH = os.getenv("HOME_PATH")
DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
os.chdir(HOME_PATH)

import pandas as pd
import numpy as np
import datetime
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums


def plot():
    df_sorted = df.sort_values(['subba', 'period'])

    # Create the figure using Plotly Express
    fig = px.line(df_sorted, x='period', y='value', color='subba',
                  title="Energy Consumption Time Series by Group",
                  labels={'period': 'Date', 'value': 'Energy Consumption', 'subba': 'Groups'},
                  height=600, width=1000)

    # Update layout
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True] * len(df_sorted['subba'].unique())},
                               {"title": "Energy Consumption Time Series - All sub-regions"}]),
                    *[dict(label=subba,
                           method="update",
                           args=[{"visible": [subba == s for s in df_sorted['subba'].unique()]},
                                 {"title": f"Energy Consumption Time Series - {subba}"}])
                      for subba in df_sorted['subba'].unique()]
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.01,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        legend_title=" ",
        xaxis_title="Date",
        yaxis_title="Energy Consumption"
    )

    # Save the figure as an HTML file and automatically open it
    output_file = "energy_consumption_analysis_by_group.html"
    fig.write_html(output_file, auto_open=True)


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)
    df = (df.pipe(complete_timeframe, bfill=True)
            .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
            .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
            .pipe(create_date_colums, 'period')
         )

    plot()
