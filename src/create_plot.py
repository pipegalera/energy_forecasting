import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
load_dotenv()
import os
DATA_PATH = os.getenv("DATA_PATH")
HOME_PATH = os.getenv("HOME_PATH")

def main(data):

    df = data.copy()
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
                    hovertemplate='%{y:.2f} MWh'
            ),
            go.Scatter(x=filtered_df.period,
                    y=filtered_df.forecasted_value,
                    mode='lines',
                    name='Demand Forecast',
                    line=dict(color='rgb(69,123,157)', dash='dash', width=2),
                    hovertemplate='%{y:.2f} MWh'
            )
        ]

        # Compute residuals and standard deviation
        residuals = filtered_df["forecasted_value"] - filtered_df["value"]
        std_dev = np.std(residuals)

        # Create prediction intervals
        upper_bound = filtered_df["forecasted_value"] + 3 * std_dev
        lower_bound = filtered_df["forecasted_value"] - 3 * std_dev

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
        args=[{'visible': [True if subba_options[j] == subba else False for j in range(len(subba_options)) for _ in range(3)]},
            {'title': f"California Demand for Electricity Forecast - Region: {subba}"}],
        label=subba,
        method='update'
    ) for subba in subba_options]

    # Set initial visibility
    for i, subba in enumerate(subba_options):
        is_visible = subba == "PGAE"
        fig.data[i*3].visible = is_visible
        fig.data[i*3+1].visible = is_visible
        fig.data[i*3+2].visible = is_visible

    fig.update_layout(
        height=500,
        font=dict(
            family='"Open Sans", verdana, arial, sans-serif',
            size=12
        ),
        title=f"California Demand for Electricity Forecast - Region: PGAE",
        xaxis_title="",
        yaxis_title="Megawatts-Hour",
        template='plotly_white',
        showlegend=True,
        hovermode="x unified",
        hoverlabel=dict(
            font=dict(
                family='"Open Sans", verdana, arial, sans-serif',
                size=12
            ),
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
            spikemode="across",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family='"Open Sans", verdana, arial, sans-serif'),
        ),
        # Add rangeslider and rangeselector here
        xaxis_rangeslider_visible=True,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 12, "t": 12},
                showactive=True,
                x=0,
                xanchor="left",
                y=1,
                yanchor="bottom",
                bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="rgba(69,123,157,0.2)",
                borderwidth=1,
                font=dict(
                    family='"Open Sans", verdana, arial, sans-serif',
                    size=12
                ),
            )
        ],
    )

    return fig

if __name__=="__main__":
    df = pd.read_parquet(f"{DATA_PATH}/inference.parquet")
    fig = main(df)
    fig.write_html(f"{HOME_PATH}/docs/plot.html")
    print("--> Visualization updated!")
