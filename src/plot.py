import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



df = pd.read_parquet("data/inference.parquet")
df = df[['period', 'subba', 'value', 'forecasted_value']].sort_values(by=["subba","period"])
cutoff_date = df["period"].iloc[-1] - pd.Timedelta(days=50)


subba_options = df['subba'].unique().tolist()
df = df[df["period"] > cutoff_date]


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Function to create traces for a given subba
def get_traces(selected_subba):
    filtered_df = df[df['subba'] == selected_subba]

    traces = [
        go.Scatter(x=filtered_df.period, y=filtered_df.value, mode='lines', name='Demand', line=dict(color='teal')),
        go.Scatter(x=filtered_df.period, y=filtered_df.forecasted_value, mode='lines', name='Demand Forecast', line=dict(dash='dash', color='blue'))
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
        fillcolor='rgba(173, 216, 230, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Forecast Error Bands'
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

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=dropdown_buttons,
        x=1,
        y=1.3,
        xanchor='left',
        yanchor='top'
    )]
)

# Set initial visibility
fig.data[0].visible = True
fig.data[1].visible = True
fig.data[2].visible = True
for i in range(3, len(fig.data)):
    fig.data[i].visible = False

# Update layout
fig.update_layout(
    title=f"California Demand for Electricity Forecast - Region: {subba_options[0]}",
    xaxis_title="",
    yaxis_title="Electricity Demand (MWh)",
    template='plotly_white',
    showlegend=True
)

# Add MAPE and Coverage as annotations (you'll need to calculate these for each subba)
#fig.add_annotation(
#    xref="paper", yref="paper",
#    x=1.14, y=0.4,
#    text=f"MAPE: {MAPE}%<br>Coverage: {coverage}%",
#    align="left",
#    showarrow=False
#)

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
