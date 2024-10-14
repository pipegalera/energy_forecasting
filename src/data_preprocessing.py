from dotenv import load_dotenv
load_dotenv()
import sys
import os
DATA_PATH = os.getenv("DATA_PATH")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils import complete_timeframe, create_group_lags, create_group_rolling_means, create_date_colums, create_horizon


if __name__=="__main__":
    print("Reading raw data.parquet...")
    df = pd.read_parquet(f"{DATA_PATH}/data.parquet")
    df = (df.pipe(complete_timeframe, bfill=True)
            #.pipe(create_horizon, 'subba', horizon_days=60)
            .pipe(create_group_lags, 'subba', ['value'], lags=[3,6,12,24,48,168,336,720,2160])
            .pipe(create_group_rolling_means, 'subba', ['value'], windows=[3,6,12,24,48,168,336,720,2160])
            .pipe(create_date_colums, 'period')
         )
    df = df.sort_values(['subba', 'period'])
    print("Preprocessing data...")
    df.to_parquet(f"{DATA_PATH}/data_preprocessed.parquet")
    print("Done! New data_preprocessed.parquet file updated")
