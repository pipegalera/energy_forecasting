from data_backfill import eia_backfill_data
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

def run_refresh():
    old_data = pd.read_parquet("./data/data.parquet")
    print("--> Rows before:",  f"{len(old_data):,}")
    print("--> Current data updated until:", old_data.period.max())

    EIA_API_KEY = os.getenv('EIA_API_KEY')
    API_PATH = "electricity/rto/region-sub-ba-data/"
    facets = {
        'parent': ['CISO'],
        'subba': ['PGAE', 'SCE', 'SDGE', 'VEA'],
    }
    last_date = pd.to_datetime(old_data['period'].max())
    start = (last_date + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    end = datetime.utcnow()

    new_data = eia_backfill_data(api_key=EIA_API_KEY,
                            api_path=API_PATH,
                            start = start,
                            end = end,
                            length = 5000,
                            frequency = "hourly",
                            facets=facets,
                            refresh=True)

    print("--> Rows after:", f"{len(new_data):,}")
    print("--> New Data Updated until:", new_data.period.max())

if __name__=="__main__":
    run_refresh()