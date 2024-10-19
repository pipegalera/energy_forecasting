import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yaml

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

# Load text
with open('dashboard/assets/dashboard_text.yaml', 'r') as file:
    content = yaml.safe_load(file)

# Load data
df = pd.read_parquet(f"{DATA_PATH}/inference.parquet")
subba_options = df['subba'].unique().tolist()

# Initialize the Dash app
app = dash.Dash(__name__)

# Initialize the Dash app with custom stylesheet
app = dash.Dash(__name__, external_stylesheets=['/assets/custom.css'])

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1(content['title']),
        html.Div([dcc.Markdown(content["description"])],
            className='body'),
        html.Div([
            html.Label("Please select the preferred electric company for the region:"),
            dcc.Dropdown(
                id='subba-dropdown',
                options=[{'label': i, 'value': i} for i in subba_options],
                value=subba_options[0],
                className='dash-dropdown'
            )
        ], className='dropdown-container'),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=dcc.Graph(id='forecast-graph')
        ),
        html.Div([dcc.Markdown(content["footer"])],
            className='footer')
    ],
    className='container')
])

# Callback to update the graph
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('subba-dropdown', 'value')
)
def update_graph(selected_subba):
    filtered_df = df[df['subba'] == selected_subba]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add demand trace
    fig.add_trace(go.Scatter(
        x=filtered_df.period,
        y=filtered_df.value,
        mode='lines',
        name='Demand',
        line=dict(color='rgb(31,119,180)'),
        hovertemplate='%{y} MWh'
    ))

    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=filtered_df.period,
        y=filtered_df.forecasted_value,
        mode='lines',
        name='Forecast',
        line=dict(color='rgb(69,123,157)', dash='dash', width=2),
        hovertemplate='%{y} MWh'
    ))

    # Compute residuals and standard deviation
    residuals = filtered_df["forecasted_value"] - filtered_df["value"]
    std_dev = np.std(residuals)

    # Create prediction intervals
    upper_bound = filtered_df["forecasted_value"] + 3 * std_dev
    lower_bound = filtered_df["forecasted_value"] - 3 * std_dev

    # Add prediction intervals
    fig.add_trace(go.Scatter(
        x=filtered_df.period.tolist() + filtered_df.period.tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(69,123,157,0.2)',
        line=dict(color='rgba(69,123,157,0.4)'),
        hoverinfo="skip",
        showlegend=True,
        name='Forecast Intervals'
    ))

    fig.update_layout(
        title=f"Electricity Demand Forecast - {selected_subba}",
        xaxis_title="",
        yaxis_title="Megawatts-Hour",
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
            spikemode="across",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=3, label="3d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="todate"),
                    dict(step="all")
                ]),
                y=1.1,
                x=0,
                yanchor='top'
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
