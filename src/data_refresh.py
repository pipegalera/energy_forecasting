from dotenv import load_dotenv
load_dotenv()
import os
import sys

HOME_PATH = os.getenv("HOME_PATH")
os.chdir(HOME_PATH)
sys.path.append(os.getcwd())

import pandas as pd
from datetime import datetime, timedelta, timezone
from src.data_backfill import eia_backfill_data

DATA_PATH = os.getenv("DATA_PATH")
EIA_API_KEY = os.getenv('EIA_API_KEY')

def run_refresh():
    old_data = pd.read_parquet(f"{DATA_PATH}/data.parquet")
    print("--> Rows before:",  f"{len(old_data):,}")
    print("--> Current data updated until:", old_data.period.max().strftime("%Y-%m-%dT%H"))

    API_PATH = "electricity/rto/region-sub-ba-data/"
    facets = {
        'parent': ['CISO'],
        'subba': ['PGAE', 'SCE', 'SDGE', 'VEA'],
    }
    last_date = old_data['period'].max().tz_convert('UTC').strftime("%Y-%m-%dT%H")
    print(f"--> Retrieving data from: {last_date} ...")
    new_data = eia_backfill_data(api_key=EIA_API_KEY,
                            api_path=API_PATH,
                            start = last_date,
                            length = 5000,
                            frequency = "hourly",
                            facets=facets,
                            refresh=True)

    print("--> Rows after:", f"{len(new_data):,}")
    print("--> New Data Updated until:", new_data.period.max().strftime("%Y-%m-%dT%H"))
    print("---------------------------------------------")

if __name__=="__main__":
    run_refresh()
