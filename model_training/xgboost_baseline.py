from dotenv import load_dotenv
load_dotenv()
import os
import sys
import pandas as pd
import numpy as np

HOME_PATH = os.getenv("HOME_PATH")
DATA_PATH = os.getenv("DATA_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")

os.chdir(HOME_PATH)

from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums

df = pd.read_parquet(DATA_PATH + "data.parquet")

df = (df.pipe(complete_timeframe, bfill=True)
        .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
        .pipe(create_date_colums, 'period')
     )
df = df.sort_values(['subba', 'period'])

print(df.dtypes)
