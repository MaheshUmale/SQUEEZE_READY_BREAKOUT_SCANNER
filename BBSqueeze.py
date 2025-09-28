import os
import urllib.parse
import json
from time import sleep
from datetime import datetime
import numpy as np
import sqlite3
from tradingview_screener import Query, col, And, Or
import pandas as pd

# --- SQLite Timestamp Handling ---
def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 format."""
    return val.isoformat()

def convert_timestamp(val):
    """Convert ISO 8601 string to datetime.datetime object."""
    return datetime.fromisoformat(val.decode())

# Register the adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp)


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
    required_cols = ['ticker', 'HeatmapScore', 'SqueezeCount', 'rvol', 'URL', 'logo', 'momentum', 'highest_tf', 'squeeze_strength']
    for c in required_cols:
        if c not in df.columns:
            # Provide a default value if a column is missing
            if c == 'momentum':
                df[c] = 'Neutral'
            elif c == 'highest_tf' or c == 'squeeze_strength':
                df[c] = 'N/A'
            else:
                df[c] = 0

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
            "momentum": row['momentum'],
            "highest_tf": row['highest_tf'],
            "squeeze_strength": row['squeeze_strength']
        })

    # Write the JSON file
    with open(output_path, 'w') as f:
        json.dump(heatmap_data, f, indent=4)
    print(f"✅ Flat JSON successfully generated at '{output_path}'.")


# --- Configuration ---
DB_FILE = 'squeeze_history.db'
OUTPUT_JSON_FIRED = 'treemap_data_fired.json'
OUTPUT_JSON_IN_SQUEEZE = 'treemap_data_in_squeeze.json'
TIME_INTERVAL_SECONDS = 120

# --- Timeframe Configuration ---
timeframes = ['', '|1', '|5', '|15', '|30', '|60', '|120', '|240', '|1W', '|1M']
tf_order_map = {
    '|1M': 10, '|1W': 9, '|240': 8, '|120': 7, '|60': 6,
    '|30': 5, '|15': 4, '|5': 3, '|1': 2, '': 1
}
tf_display_map = {
    '': 'Daily', '|1': '1m', '|5': '5m', '|15': '15m', '|30': '30m',
    '|60': '1H', '|120': '2H', '|240': '4H', '|1W': 'Weekly', '|1M': 'Monthly'
}
tf_suffix_map = {v: k for k, v in tf_display_map.items()}

# Construct select columns for all timeframes
select_cols = [
    'name', 'logoid', 'close', 'volume|5', 'Value.Traded|5', 'average_volume_10d_calc|5', 'MACD.hist'
]
for tf in timeframes:
    select_cols.extend([f'KltChnl.lower{tf}', f'KltChnl.upper{tf}', f'BB.lower{tf}', f'BB.upper{tf}', f'SMA20{tf}', f'ATR{tf}'])


def get_highest_squeeze_tf(row):
    for tf_suffix in sorted(tf_order_map, key=tf_order_map.get, reverse=True):
        if row.get(f'InSqueeze{tf_suffix}', False):
            return tf_display_map[tf_suffix]
    return 'Unknown'

def get_squeeze_strength(row):
    """Calculates and categorizes the strength of the squeeze."""
    highest_tf_name = row['highest_tf']
    tf_suffix = tf_suffix_map.get(highest_tf_name)

    if tf_suffix is None:
        return "N/A"

    bb_upper = row.get(f'BB.upper{tf_suffix}')
    bb_lower = row.get(f'BB.lower{tf_suffix}')
    kc_upper = row.get(f'KltChnl.upper{tf_suffix}')
    kc_lower = row.get(f'KltChnl.lower{tf_suffix}')

    if any(pd.isna(val) for val in [bb_upper, bb_lower, kc_upper, kc_lower]):
        return "N/A"

    bb_width = bb_upper - bb_lower
    kc_width = kc_upper - kc_lower

    if bb_width == 0:
        return "N/A" # Avoid division by zero

    sqz_strength = kc_width / bb_width

    if sqz_strength >= 2:
        return "VERY STRONG"
    elif sqz_strength >= 1.5:
        return "STRONG"
    elif sqz_strength > 1:
        return "Regular"
    else:
        return "N/A"


def calculate_volatility(row, tf_suffix):
    """Calculates volatility for a given timeframe."""
    bb_upper = row.get(f'BB.upper{tf_suffix}')
    sma20 = row.get(f'SMA20{tf_suffix}')
    atr = row.get(f'ATR{tf_suffix}')

    if any(pd.isna(val) for val in [bb_upper, sma20, atr]) or atr == 0:
        return 0.0 # Return a default float value

    price_std_deviation = (bb_upper - sma20) / 2
    volatility = price_std_deviation / atr
    return volatility


def init_db():
    """Initializes the database and creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check for schema changes
    try:
        cursor.execute("PRAGMA table_info(squeeze_history)")
        columns = {col[1]: col[2] for col in cursor.fetchall()}
        if 'volatility' not in columns or 'timeframe' not in columns or columns.get('scan_timestamp') != 'TIMESTAMP':
             print("Schema outdated. Recreating squeeze_history table.")
             cursor.execute("DROP TABLE IF EXISTS squeeze_history")
    except sqlite3.OperationalError:
        # Table doesn't exist, will be created.
        pass

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS squeeze_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_timestamp TIMESTAMP NOT NULL,
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            volatility REAL NOT NULL
        )
    ''')
    # Create an index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON squeeze_history (scan_timestamp)')
    conn.commit()
    conn.close()


def load_previous_squeeze_list_from_db():
    """Loads the list of (ticker, timeframe) tuples from the most recent scan."""
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    cursor.execute('SELECT MAX(scan_timestamp) FROM squeeze_history')
    last_timestamp = cursor.fetchone()[0]

    if last_timestamp is None:
        conn.close()
        return []

    cursor.execute('SELECT ticker, timeframe FROM squeeze_history WHERE scan_timestamp = ?', (last_timestamp,))
    squeeze_pairs = [(row[0], row[1]) for row in cursor.fetchall()]

    conn.close()
    return squeeze_pairs


def save_current_squeeze_list_to_db(squeeze_data):
    """Saves the current list of (ticker, timeframe, volatility) tuples to the database."""
    if not squeeze_data:
        return # Don't save if there's nothing to save

    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    now = datetime.now()

    data_to_insert = [(now, ticker, tf, vol) for ticker, tf, vol in squeeze_data]
    cursor.executemany('INSERT INTO squeeze_history (scan_timestamp, ticker, timeframe, volatility) VALUES (?, ?, ?, ?)',
                       data_to_insert)

    conn.commit()
    conn.close()


# --- Main Loop ---
init_db()
while True:
    try:
        # 1. Load the list of (ticker, timeframe) pairs that were in a squeeze previously
        prev_squeeze_pairs = load_previous_squeeze_list_from_db()
        print(f"Loaded {len(prev_squeeze_pairs)} (ticker, timeframe) pairs from the previous scan.")

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
        ).set_markets('india')

        _, df_in_squeeze = query_in_squeeze.get_scanner_data()

        current_squeeze_data = []
        if df_in_squeeze is not None and not df_in_squeeze.empty:
            print(f"Found {len(df_in_squeeze)} tickers currently in a squeeze across all timeframes.")

            # Create a list of all (ticker, timeframe, volatility) tuples currently in a squeeze
            for _, row in df_in_squeeze.iterrows():
                for tf_suffix, tf_name in tf_display_map.items():
                    is_in_squeeze = (row[f'BB.upper{tf_suffix}'] < row[f'KltChnl.upper{tf_suffix}']) and \
                                    (row[f'BB.lower{tf_suffix}'] > row[f'KltChnl.lower{tf_suffix}'])
                    if is_in_squeeze:
                        volatility = calculate_volatility(row, tf_suffix)
                        current_squeeze_data.append((row['ticker'], tf_name, volatility))

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
            df_in_squeeze['highest_tf'] = df_in_squeeze.apply(get_highest_squeeze_tf, axis=1)
            df_in_squeeze['squeeze_strength'] = df_in_squeeze.apply(get_squeeze_strength, axis=1)

            avg_vol = df_in_squeeze['average_volume_10d_calc|5'].replace(0, np.nan)
            df_in_squeeze['rvol'] = (df_in_squeeze['volume|5'] / avg_vol).fillna(0)
            df_in_squeeze['momentum'] = df_in_squeeze['MACD.hist'].apply(get_momentum_indicator)

            momentum_map = {'Bullish': 1, 'Neutral': 0.5, 'Bearish': -1}
            df_in_squeeze['HeatmapScore'] = (df_in_squeeze['rvol'] + 1) * df_in_squeeze['SqueezeCount'] * df_in_squeeze['momentum'].map(momentum_map)

            generate_heatmap_json(df_in_squeeze, OUTPUT_JSON_IN_SQUEEZE)
        else:
            print("No tickers found currently in a squeeze.")
            generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_IN_SQUEEZE)

        # 3. Identify and process "Squeeze Fired" stocks
        current_squeeze_pairs = [(ticker, tf) for ticker, tf, vol in current_squeeze_data]
        fired_pairs = set(prev_squeeze_pairs) - set(current_squeeze_pairs)
        fired_tickers = list(set(ticker for ticker, tf in fired_pairs))
        print(f"Found {len(fired_pairs)} (ticker, timeframe) pairs where squeeze has fired.")

        if fired_tickers:
            query_fired = Query().select(*select_cols).where2(col('name').is_in(fired_tickers)).set_markets('india')
            _, df_fired = query_fired.get_scanner_data()

            if df_fired is not None and not df_fired.empty:
                current_DATE = datetime.now().strftime('%d_%m_%y')
                df_fired['encodedTicker'] = df_fired['ticker'].apply(urllib.parse.quote)
                df_fired['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + df_fired['encodedTicker']
                df_fired['logo'] = df_fired['logoid'].apply(
                    lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')

                df_fired['SqueezeCount'] = 1
                avg_vol = df_fired['average_volume_10d_calc|5'].replace(0, np.nan)
                df_fired['rvol'] = (df_fired['volume|5'] / avg_vol).fillna(0)
                df_fired['momentum'] = df_fired['MACD.hist'].apply(get_momentum_indicator)
                df_fired['squeeze_strength'] = 'N/A' # Not applicable for fired squeezes
                df_fired['highest_tf'] = 'N/A' # Not applicable for fired squeezes

                momentum_map = {'Bullish': 1, 'Neutral': 0.5, 'Bearish': -1}
                df_fired['HeatmapScore'] = (df_fired['rvol'] + 1) * df_fired['SqueezeCount'] * df_fired['momentum'].map(momentum_map)

                generate_heatmap_json(df_fired, OUTPUT_JSON_FIRED)
                append_df_to_csv(df_fired, 'BBSCAN_FIRED_' + str(current_DATE) + '.csv')
                print("--- Fired Squeeze Results ---")
                print(df_fired[['ticker', 'momentum', 'SqueezeCount', 'rvol', 'HeatmapScore']])
            else:
                generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_FIRED)
        else:
            print("No squeezes fired in this interval.")
            generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_FIRED)

        # 4. Save the current list of squeezed stocks for the next iteration
        save_current_squeeze_list_to_db(current_squeeze_data)
        print(f"Saved {len(current_squeeze_data)} (ticker, timeframe, volatility) tuples for the next scan.")

    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"\n--- Waiting for {TIME_INTERVAL_SECONDS} seconds until the next scan ---\n")
    sleep(TIME_INTERVAL_SECONDS)