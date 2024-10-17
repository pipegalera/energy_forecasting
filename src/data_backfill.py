import requests
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
EIA_API_KEY = os.getenv('EIA_API_KEY')
API_PATH = "electricity/rto/region-sub-ba-data/"

facets = {
    'parent': ['CISO'],
    'subba': ['PGAE', 'SCE', 'SDGE', 'VEA'],
}

def format_facets(facets_dict):
    formatted_facets = {}
    for facet_type, values in facets_dict.items():
        formatted_facets[f'facets[{facet_type}][]'] = values
    return formatted_facets

def eia_get_data(api_key,
                 api_path,
                 start=None,
                 end=None,
                 length=None,
                 offset=0,
                 frequency=None,
                 facets=None):

    start = pd.to_datetime(start, utc=True).strftime("%Y-%m-%dT%H")

    api_path = api_path if api_path.endswith("/") else api_path + "/"
    url = "https://api.eia.gov/v2/" + api_path + 'data/'

    parameters = {
        'frequency': frequency,
        'data[0]': 'value',
        'start': start,
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'offset': offset,
        'length': length,
        'api_key': api_key,
    }

    if facets:
        parameters.update(format_facets(facets))

    #print(requests.get(url, params=parameters).url)
    #print(requests.get(url, params=parameters))

    try:
        response = requests.get(url, params=parameters).json()
        df = pd.DataFrame(response['response']['data'])
        df["value"] = pd.to_numeric(df["value"])
        df["period"] = pd.to_datetime(df["period"], utc=True)
        df = df[df["period"] > start] # For safety, the api retrieves data lazily (without caring the hours)
        df = df.sort_values(by=[df.columns[0], df.columns[3], df.columns[1]])
        return df
    except:
        return None


def eia_backfill_data(api_key,
                api_path,
                start = None,
                length = None,
                offset = 0,
                frequency = None,
                facets = None,
                refresh = False):

    all_data = []
    offset = 0
    while True:
        df = eia_get_data(api_key = api_key,
                api_path = api_path,
                start = start,
                length = length,
                frequency = frequency,
                facets = facets,
                offset = offset)

        if df is None or df.empty:
            break

        all_data.append(df)
        offset += length

        print(f"Retrieved {len(df)} datapoints.")

        # Print the last row of the most recently appended data
        # print("Last row of the most recent data:")
        # print(df.iloc[-1])
        # print("\n")

    if refresh:
        if not all_data:
            print("--> No new data found in the API")
            df = pd.read_parquet(f"{DATA_PATH}/data.parquet")
        else: # else = there is new data
            df = pd.concat([
                pd.read_parquet(f"{DATA_PATH}/data.parquet"),
                pd.concat(all_data, ignore_index=True)
            ]).sort_values(by=["subba","period"])
            df.to_parquet(f"{DATA_PATH}/data.parquet", index=False)
    else: # else = new backfill
        df = pd.concat(all_data, ignore_index=True).sort_values(by=["subba","period"])
        df.to_parquet(f"{DATA_PATH}/data.parquet", index=False)
        print(f"Total datapoints writen so far: {sum(len(d) for d in all_data)}")

    return df

if __name__== "__main__":
    # Backfill of data
    eia_backfill_data(api_key=EIA_API_KEY,
                            api_path=API_PATH,
                            start = '2019-01-01',
                            length = 5000,
                            frequency = "hourly",
                            facets=facets,
                            refresh=False)
