import requests
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

EIA_API_KEY = os.getenv('EIA_API_KEY')
DATA_PATH = "/Users/pipegalera/dev/Documents/GitHub/EIA_tracker/data/data.parquet"
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
            start = None,
            end = None,
            length = None,
            offset = 0,
            frequency = None,
            facets = None):


    start = pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S")
    end   = pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S")
    api_path = api_path if api_path.endswith("/") else api_path + "/"
    url = "https://api.eia.gov/v2/" + api_path + 'data/'

    parameters = {
        'api_key': api_key,
        'frequency': frequency,
        'offset': offset,
        'data[]': 'value',
        'start': start,
        'end': end,
        'length': length,
    }

    if facets:
        parameters = {**parameters, **format_facets(facets)}

    # print(requests.get(url, params=parameters).url)
    # print(requests.get(url, params=parameters))

    response = requests.get(url, params=parameters).json()


    if len(response) <= 2:
    # len=2 == {'error': 'Call to a member function format() on bool', 'code': 0}
        return None
    else:
        df = pd.DataFrame(response['response']['data'])
        df["value"] = pd.to_numeric(df["value"])
        df = df.sort_values(by = [df.columns[0],df.columns[3], df.columns[1]])
        return df


def eia_backfill_data(api_key,
                api_path,
                start = None,
                end = None,
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
                              end = end,
                              length = length,
                              frequency = frequency,
                              facets = facets,
                              offset = offset)

            if df is None or df.empty:
                        break

            all_data.append(df)
            offset += length

            print(f"Retrieved {len(df)} rows. Total rows so far: {sum(len(d) for d in all_data)}")

            # Print the last row of the most recently appended data
            # print("Last row of the most recent data:")
            # print(df.iloc[-1])
            # print("\n")

    if refresh:
        if not all_data:
            print("--> No new data found in the API")
            df = pd.read_parquet("./data/data.parquet")
        else:
            df = pd.concat([
                pd.read_parquet("./data/data.parquet"),
                pd.concat(all_data, ignore_index=True)
            ])
            df.to_parquet("./data/data.parquet", index=False)
    else: # else = new backfill
        df = pd.concat(all_data, ignore_index=True)

    #print("Data saved under:", DATA_PATH)

    return df

if __name__== "__main__":
    # Backfill of data
    eia_backfill_data(api_key=EIA_API_KEY,
                            api_path=API_PATH,
                            start = '2019-01-01',
                            end = '2024-01-01',
                            length = 5000,
                            frequency = "hourly",
                            facets=facets,
                            refresh=False)
