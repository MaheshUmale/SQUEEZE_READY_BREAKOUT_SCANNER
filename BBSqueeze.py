import os
import urllib.parse
import json
from time import sleep
from datetime import datetime
import numpy as np
from tradingview_screener import Query, col, And, Or
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)


def append_df_to_csv(df, csv_path):
    """
    Appends a DataFrame to a CSV file. Creates the file with a header if it doesn't
    exist, otherwise appends without the header.
    """
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


def get_momentum_indicator(macd_hist_value):
    """
    Determines momentum based on the MACD histogram value.
    """
    if macd_hist_value > 0:
        return 'Bullish'
    elif macd_hist_value < 0:
        return 'Bearish'
    else:
        return 'Neutral'


def generate_heatmap_json(df, output_path):
    """
    Generates a simple, flat JSON array of stock data for the D3 heatmap.
    """
    # Ensure required columns exist for JSON generation
    required_cols = ['ticker', 'HeatmapScore', 'SqueezeCount', 'rvol', 'URL', 'logo', 'momentum']
    for c in required_cols:
        if c not in df.columns:
            # Provide a default value if a column is missing
            if c == 'momentum':
                df[c] = 'Neutral'
            else:
                df[c] = 0 if 'Score' in c or 'Count' in c or 'rvol' in c else ''

    # Create a flat list of stock data
    heatmap_data = []
    for _, row in df.iterrows():
        heatmap_data.append({
            "name": row['ticker'],
            "value": row['HeatmapScore'],
            "count": row['SqueezeCount'],
            "rvol": row['rvol'],
            "url": row['URL'],
            "logo": row['logo'],
            "momentum": row['momentum']
        })

    # Write the JSON file
    with open(output_path, 'w') as f:
        json.dump(heatmap_data, f, indent=4)
    print(f"✅ Flat JSON successfully generated at '{output_path}'.")


# --- Configuration ---
PREV_SQUEEZE_FILE = 'previous_squeeze_list.json'
OUTPUT_JSON_FIRED = 'treemap_data_fired.json'
OUTPUT_JSON_IN_SQUEEZE = 'treemap_data_in_squeeze.json'
TIME_INTERVAL_SECONDS = 120

# Timeframes to check for a squeeze
timeframes = ['', '|1', '|5', '|15', '|30', '|60', '|120', '|240', '|1W', '|1M']

# Construct select columns for all timeframes
select_cols = [
    'name', 'logoid', 'close', 'volume|5', 'Value.Traded|5', 'average_volume_10d_calc|5', 'MACD.macd_hist'
]
for tf in timeframes:
    select_cols.extend([f'KltChnl.lower{tf}', f'KltChnl.upper{tf}', f'BB.lower{tf}', f'BB.upper{tf}'])


def load_previous_squeeze_list(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def save_current_squeeze_list(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)


# --- Main Loop ---
while True:
    try:
        # 1. Load the list of tickers that were in a squeeze previously
        prev_squeeze_list = load_previous_squeeze_list(PREV_SQUEEZE_FILE)
        print(f"Loaded {len(prev_squeeze_list)} tickers from the previous scan.")

        # 2. Find all stocks currently in a squeeze (fetch full data)
        squeeze_conditions = [
            And(
                col(f'BB.upper{tf}') < col(f'KltChnl.upper{tf}'),
                col(f'BB.lower{tf}') > col(f'KltChnl.lower{tf}')
            ) for tf in timeframes
        ]
        query_in_squeeze = Query().select(*select_cols).where2(
            And(
                col('is_primary') == True, col('typespecs').has('common'), col('type') == 'stock',
                col('exchange') == 'NSE', col('close').between(20, 10000), col('active_symbol') == True,
                col('average_volume_10d_calc|5') > 50000, col('Value.Traded|5') > 10000000,
                Or(*squeeze_conditions)
            )
        ).limit(500).set_markets('india')

        _, df_in_squeeze = query_in_squeeze.get_scanner_data()

        current_squeeze_list = []
        if df_in_squeeze is not None and not df_in_squeeze.empty:
            print(f"Found {len(df_in_squeeze)} tickers currently in a squeeze.")
            current_squeeze_list = df_in_squeeze['ticker'].tolist()

            # --- Process and save "In Squeeze" data ---
            df_in_squeeze['encodedTicker'] = df_in_squeeze['ticker'].apply(urllib.parse.quote)
            df_in_squeeze['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + df_in_squeeze['encodedTicker']
            df_in_squeeze['logo'] = df_in_squeeze['logoid'].apply(
                lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')

            squeeze_count_cols = []
            for tf in timeframes:
                col_name = f'InSqueeze{tf}'
                df_in_squeeze[col_name] = (df_in_squeeze[f'BB.upper{tf}'] < df_in_squeeze[f'KltChnl.upper{tf}']) & \
                                          (df_in_squeeze[f'BB.lower{tf}'] > df_in_squeeze[f'KltChnl.lower{tf}'])
                squeeze_count_cols.append(col_name)
            df_in_squeeze['SqueezeCount'] = df_in_squeeze[squeeze_count_cols].sum(axis=1)

            avg_vol = df_in_squeeze['average_volume_10d_calc|5'].replace(0, np.nan)
            df_in_squeeze['rvol'] = (df_in_squeeze['volume|5'] / avg_vol).fillna(0)
            df_in_squeeze['HeatmapScore'] = (df_in_squeeze['rvol'] + 1) * df_in_squeeze['SqueezeCount']
            df_in_squeeze['momentum'] = 'Neutral' # Momentum is not applicable for "in squeeze"

            generate_heatmap_json(df_in_squeeze, OUTPUT_JSON_IN_SQUEEZE)
        else:
            print("No tickers found currently in a squeeze.")
            # Write empty JSON if no stocks are in a squeeze
            generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_IN_SQUEEZE)

        # 3. Identify and process "Squeeze Fired" stocks
        fired_tickers = [ticker for ticker in prev_squeeze_list if ticker not in current_squeeze_list]
        print(f"Found {len(fired_tickers)} tickers where squeeze has fired.")

        if fired_tickers:
            # We need to re-query to get the latest data at the moment of firing
            query_fired = Query().select(*select_cols).where2(col('name').is_in(fired_tickers)).set_markets('india')
            _, df_fired = query_fired.get_scanner_data()

            if df_fired is not None and not df_fired.empty:
                current_DATE = datetime.now().strftime('%d_%m_%y')
                df_fired['encodedTicker'] = df_fired['ticker'].apply(urllib.parse.quote)
                df_fired['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + df_fired['encodedTicker']
                df_fired['logo'] = df_fired['logoid'].apply(
                    lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')

                # We assume it was in a squeeze on at least one timeframe.
                df_fired['SqueezeCount'] = 1

                avg_vol = df_fired['average_volume_10d_calc|5'].replace(0, np.nan)
                df_fired['rvol'] = (df_fired['volume|5'] / avg_vol).fillna(0)
                df_fired['momentum'] = df_fired['MACD.macd_hist'].apply(get_momentum_indicator)
                df_fired['HeatmapScore'] = (df_fired['rvol'] + 1) * df_fired['SqueezeCount']

                generate_heatmap_json(df_fired, OUTPUT_JSON_FIRED)
                append_df_to_csv(df_fired, 'BBSCAN_FIRED_' + str(current_DATE) + '.csv')
                print("--- Fired Squeeze Results ---")
                print(df_fired[['ticker', 'momentum', 'SqueezeCount', 'rvol', 'HeatmapScore']])
            else:
                # Write empty JSON if fired tickers couldn't be fetched
                generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_FIRED)
        else:
            print("No squeezes fired in this interval.")
            # Write empty JSON if no stocks have fired
            generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_FIRED)

        # 4. Save the current list of squeezed stocks for the next iteration
        save_current_squeeze_list(current_squeeze_list, PREV_SQUEEZE_FILE)
        print(f"Saved {len(current_squeeze_list)} currently squeezed tickers for the next scan.")

    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"\n--- Waiting for {TIME_INTERVAL_SECONDS} seconds until the next scan ---\n")
    sleep(TIME_INTERVAL_SECONDS)

