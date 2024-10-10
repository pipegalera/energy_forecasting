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
from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums, create_horizon

import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_parquet(DATA_PATH)
df = (df.pipe(complete_timeframe, bfill=True)
        .pipe(create_horizon, 'subba', horizon_days=60)
        .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_date_colums, 'period')
     )


reg_new = xgb.XGBRegressor()
reg_new.load_model(MODELS_PATH + "/xgb/xgb_model_PGAE.json")


target='value',
covs= covs = ['period_day',
        'period_week', 'period_year',
        'period_day_of_week','period_day_of_year',
        'period_month_end', 'period_month_start',
        'period_year_end', 'period_year_start',
        'period_quarter_end', 'period_quarter_start',]




pgae_values = df[(df["subba"] == "PGAE") & (df["data_type"] == "Real values")].set_index("period").sort_index()
pgae_preds = df[(df["subba"] == "PGAE") & (df["data_type"] == "Predicted values")].set_index("period").sort_index()
pgae_preds["values"] = reg_new.predict(pgae_preds[covs])


import plotly.express as px

fig = px.line()
fig.add_scatter(x=pgae_values.index, y=pgae_values["value"], name="Real Values")
fig.add_scatter(x=pgae_preds.index, y=pgae_preds["values"], name="Predicted Values")
fig.add_hline(y=pgae_average, line_dash="dash", line_color="green", annotation_text="Average", annotation_position="bottom right")
fig.update_layout(title="Real Values and Future Predictions")
output_file = "energy_consumption_analysis_by_group.html"
fig.write_html(output_file, auto_open=True)


import plotly.express as px
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np

combined_df = pd.concat([
        pgae_values["value"].rename("value"),
        pgae_preds["values"].rename("value")
    ]).sort_index().reset_index()


# Convert index to numeric for LOWESS
combined_df['numeric_index'] = (combined_df.index - combined_df.index.min())/ 3600
combined_df

# Apply LOWESS smoothing
smoothed = lowess(combined_df["value"], combined_df["numeric_index"], frac=0.3)

# Create the figure
fig = go.Figure()

# Add real values
fig.add_trace(go.Scatter(x=pgae_values.index, y=pgae_values["value"], name="Real Values", mode="lines"))

# Add predicted values
fig.add_trace(go.Scatter(x=pgae_preds.index, y=pgae_preds["values"], name="Predicted Values", mode="lines"))

# Add smoothed line
smoothed_dates = combined_df.index.min() + pd.to_timedelta(smoothed[:, 0], unit='D')
fig.add_trace(go.Scatter(x=smoothed_dates, y=smoothed[:, 1], name="Smoothed Trend", line=dict(color='red', width=2)))

# Update layout
fig.update_layout(title="Real Values, Future Predictions, and Smoothed Trend")

# Save and open the file
output_file = "energy_consumption_analysis_by_group.html"
fig.write_html(output_file, auto_open=True)
