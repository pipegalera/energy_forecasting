import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
load_dotenv()
import os
DATA_PATH = os.getenv("DATA_PATH")

df = pd.read_parquet(f"{DATA_PATH}/inference.parquet")
subba_options = df['subba'].unique().tolist()



# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Function to create traces for a given subba
def get_traces(selected_subba):
    filtered_df = df[df['subba'] == selected_subba]

    traces = [
        go.Scatter(x=filtered_df.period,
                   y=filtered_df.value,
                   mode='lines',
                   name='Demand',
                   line=dict(color='rgb(31,119,180)'),
                   hovertemplate='Demand: %{y:.2f} MWh'
        ),
        go.Scatter(x=filtered_df.period,
                   y=filtered_df.forecasted_value,
                   mode='lines',
                   name='Demand Forecast',
                   line=dict(color='rgb(69,123,157)', dash='dash', width=2),
                   hovertemplate='Forecast: %{y:.2f} MWh'
        )
    ]

    # Compute residuals and standard deviation
    residuals = filtered_df["forecasted_value"] - filtered_df["value"]
    std_dev = np.std(residuals)

    # Create prediction intervals
    upper_bound = filtered_df["forecasted_value"] + 1 * std_dev
    lower_bound = filtered_df["forecasted_value"] - 1 * std_dev

    traces.append(go.Scatter(
        x=filtered_df.period.tolist() + filtered_df.period.tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(69,123,157,0.2)',
        line=dict(color='rgba(69,123,157,0.4)'),
        hoverinfo="skip",
        showlegend=True,
        name='Prediction Intervals'
    ))

    return traces

# Add traces for each subba
for subba in subba_options:
    traces = get_traces(subba)
    for trace in traces:
        fig.add_trace(trace, secondary_y=False)

# Create and add dropdown
dropdown_buttons = [dict(
    args=[{'visible': [True if i == j else False for j in range(len(subba_options)) for _ in range(3)]},
          {'title': f"California Demand for Electricity Forecast - Region: {subba}"}],
    label=subba,
    method='update'
) for i, subba in enumerate(subba_options)]


# Set initial visibility
fig.data[0].visible = True
fig.data[1].visible = True
fig.data[2].visible = True
for i in range(3, len(fig.data)):
    fig.data[i].visible = False

fig.update_layout(
    title=f"California Demand for Electricity Forecast - Region: {subba_options[0]}",
    xaxis_title="",
    yaxis_title="Electricity Demand (MWh)",
    template='plotly_white',
    showlegend=True,
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
    ),
    hoverdistance=100,
    spikedistance=1000,
    xaxis=dict(
        showspikes=True,
        spikethickness=2,
        spikedash="dot",
        spikecolor="#999999",
        spikemode="across"
    ),
    legend=dict(
            traceorder="normal",
            font=dict(size=12),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 12, "t": 12},
                showactive=True,
                x=1.05,
                xanchor="left",
                y=1.1,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="rgba(69,123,157,0.2)",
                borderwidth=1,
            )
        ]
)

# Add source annotation
fig.add_annotation(
    xref="paper", yref="paper",
    x=-0.01, y=-0.35,
    text="<b>Data Source</b>: U.S. Energy Information Administration - EIA - Independent Statistics and Analysis",
    showarrow=False
)

# Add Model annotation
fig.add_annotation(
    xref="paper", yref="paper",
    x=-0.01, y=-0.45,
    text="<b>Forecasting</b>: XGBoost. Forecast error bands based on 1 standard error from the residuals.",
    showarrow=False
)

fig.write_html("index.html")
