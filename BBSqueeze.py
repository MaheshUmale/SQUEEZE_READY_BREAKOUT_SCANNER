from time import sleep
from datetime import datetime
from tradingview_screener import Query, col, And,Or
# Set display options to prevent column wrapping
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0) # Adjust as needed, 0 will use the full available width

from colorama import Fore, Back, Style
import urllib.parse



from collections import defaultdict
import csv


import pandas as pd
import numpy as np


master_combined =  pd.DataFrame()

import pandas as pd
import os
import json

def append_df_to_csv(df, csv_path):
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def generate_heatmap_json(df, output_path='treemap_data.json'):
    """
    Generates a hierarchical JSON file from the DataFrame for the D3 heatmap.
    """
    # Ensure required columns exist
    required_cols = ['ticker', 'HeatmapScore', 'TotalSqueezeFiredCount', 'Momentum', 'rvol', 'URL']
    if not all(col in df.columns for col in required_cols):
        print("⚠️ Cannot generate JSON. DataFrame is missing one or more required columns.")
        return

    # Create the hierarchical structure
    heatmap_data = {"name": "Squeeze Stocks", "children": []}

    # Group by momentum
    grouped = df.groupby('Momentum')

    momentum_map = {
        'Bullish': 'Bullish Momentum',
        'Bearish': 'Bearish Momentum',
        'Neutral': 'Neutral / Cross'
    }

    for group_name, group_df in grouped:
        if group_name not in momentum_map:
            continue

        group_dict = {
            "name": momentum_map[group_name],
            "children": []
        }

        for _, row in group_df.iterrows():
            group_dict["children"].append({
                "name": row['ticker'],
                "value": row['HeatmapScore'],
                "count": row['TotalSqueezeFiredCount'],
                "momentum": row['Momentum'], # This might be redundant but good for debugging
                "rvol": row['rvol'],
                "url": row['URL']
            })

        heatmap_data["children"].append(group_dict)

    # Write the JSON file
    with open(output_path, 'w') as f:
        json.dump(heatmap_data, f, indent=4)
    print(f"✅ Heatmap JSON successfully generated at '{output_path}'.")


def identify_squeeze_fired_across_timeframes(df):
    """
    Identifies when a "squeeze has fired" across all available timeframes and determines
    the breakout momentum.

    A squeeze has FIRED if:
    - The PREVIOUS candle was in a squeeze.
    - The CURRENT candle is NOT in a squeeze.

    Momentum is determined by the price relative to the Bollinger Bands on the timeframe
    where the squeeze fired.

    Args:
        df (pd.DataFrame): DataFrame with current and previous indicator values.

    Returns:
        pd.DataFrame: The DataFrame with analysis columns including 'Momentum'.
    """
    timeframes = [''] + ['1', '5', '15', '30', '60', '120', '240', '1W', '1M']

    if not all(c in df.columns for c in ['KltChnl.upper', 'BB.upper', 'KltChnl.upper[1]', 'BB.upper[1]', 'close']):
        print("Warning: Missing core columns for squeeze and momentum logic.")
        return df

    print(f"--- Identifying Squeeze Fired and Momentum for {len(timeframes)} Timeframes ---")

    squeeze_fired_cols = []
    bullish_breakout_cols = []
    bearish_breakout_cols = []

    for tf in timeframes:
        suffix = f"|{tf}" if tf else ""

        # Define columns for CURRENT and PREVIOUS candles
        kc_upper_curr, kc_lower_curr = f'KltChnl.upper{suffix}', f'KltChnl.lower{suffix}'
        bb_upper_curr, bb_lower_curr = f'BB.upper{suffix}', f'BB.lower{suffix}'
        kc_upper_prev, kc_lower_prev = f'KltChnl.upper{suffix}[1]', f'KltChnl.lower{suffix}[1]'
        bb_upper_prev, bb_lower_prev = f'BB.upper{suffix}[1]', f'BB.lower{suffix}[1]'

        # Column names for new signals
        squeeze_fired_col = f'SqueezeFired{suffix}'
        bullish_col = f'Bullish_Breakout{suffix}'
        bearish_col = f'Bearish_Breakout{suffix}'

        required = [
            kc_upper_curr, kc_lower_curr, bb_upper_curr, bb_lower_curr,
            kc_upper_prev, kc_lower_prev, bb_upper_prev, bb_lower_prev
        ]

        if all(c in df.columns for c in required):
            # Squeeze state for PREVIOUS candle
            squeeze_prev = (df[kc_upper_prev] < df[bb_upper_prev]) & (df[kc_lower_prev] > df[bb_lower_prev])
            # Squeeze state for CURRENT candle
            squeeze_curr = (df[kc_upper_curr] < df[bb_upper_curr]) & (df[kc_lower_curr] > df[bb_lower_curr])

            # Identify if the squeeze has FIRED
            df[squeeze_fired_col] = (squeeze_prev == True) & (squeeze_curr == False)
            squeeze_fired_cols.append(squeeze_fired_col)

            # Determine breakout direction for this timeframe
            df[bullish_col] = df[squeeze_fired_col] & (df['close'] > df[bb_upper_curr])
            df[bearish_col] = df[squeeze_fired_col] & (df['close'] < df[bb_lower_curr])
            bullish_breakout_cols.append(bullish_col)
            bearish_breakout_cols.append(bearish_col)

            print(f"✅ Processed Squeeze Fired and Momentum for TF '{tf}'.")
        else:
            print(f"⚠️ Skipping TF '{tf}': Missing one or more required columns.")

    # Calculate total fired squeezes
    if squeeze_fired_cols:
        df['TotalSqueezeFiredCount'] = df[squeeze_fired_cols].sum(axis=1)
        print("✅ TotalSqueezeFiredCount column added.")

    # Determine overall momentum
    if bullish_breakout_cols and bearish_breakout_cols:
        is_any_bullish = df[bullish_breakout_cols].any(axis=1)
        is_any_bearish = df[bearish_breakout_cols].any(axis=1)

        conditions = [is_any_bullish, is_any_bearish]
        choices = ['Bullish', 'Bearish']
        df['Momentum'] = np.select(conditions, choices, default='Neutral')
        print("✅ Momentum column added.")

    print("--- Processing Complete ---")
    return df


HIGH_VALUE = 50000000.00
VOLUME_MULTIPLIER =  5.0
writeHeader=True
while True :
    query = Query().select(
        'name',                    # Stock name
        'close',                   # Current price
        'volume|5',                # 1-minute volume
        'Value.Traded|5',          # 1-minute traded value
        'average_volume_10d_calc', # Average volume (10 days)
        'average_volume_10d_calc|5', # Average 1-minute volume (10 days)
        # Keltner Channel (Current and Previous)
        'KltChnl.lower', 'KltChnl.lower[1]',
        'KltChnl.lower|1', 'KltChnl.lower|1[1]',
        'KltChnl.lower|5', 'KltChnl.lower|5[1]',
        'KltChnl.lower|15', 'KltChnl.lower|15[1]',
        'KltChnl.lower|30', 'KltChnl.lower|30[1]',
        'KltChnl.lower|60', 'KltChnl.lower|60[1]',
        'KltChnl.lower|120', 'KltChnl.lower|120[1]',
        'KltChnl.lower|240', 'KltChnl.lower|240[1]',
        'KltChnl.lower|1W', 'KltChnl.lower|1W[1]',
        'KltChnl.lower|1M', 'KltChnl.lower|1M[1]',
        'KltChnl.upper', 'KltChnl.upper[1]',
        'KltChnl.upper|1', 'KltChnl.upper|1[1]',
        'KltChnl.upper|5', 'KltChnl.upper|5[1]',
        'KltChnl.upper|15', 'KltChnl.upper|15[1]',
        'KltChnl.upper|30', 'KltChnl.upper|30[1]',
        'KltChnl.upper|60', 'KltChnl.upper|60[1]',
        'KltChnl.upper|120', 'KltChnl.upper|120[1]',
        'KltChnl.upper|240', 'KltChnl.upper|240[1]',
        'KltChnl.upper|1W', 'KltChnl.upper|1W[1]',
        'KltChnl.upper|1M', 'KltChnl.upper|1M[1]',
        # Bollinger Bands (Current and Previous)
        'BB.lower', 'BB.lower[1]',
        'BB.lower|1', 'BB.lower|1[1]',
        'BB.lower|5', 'BB.lower|5[1]',
        'BB.lower|15', 'BB.lower|15[1]',
        'BB.lower|30', 'BB.lower|30[1]',
        'BB.lower|60', 'BB.lower|60[1]',
        'BB.lower|120', 'BB.lower|120[1]',
        'BB.lower|240', 'BB.lower|240[1]',
        'BB.lower|1W', 'BB.lower|1W[1]',
        'BB.lower|1M', 'BB.lower|1M[1]',
        'BB.upper', 'BB.upper[1]',
        'BB.upper|1', 'BB.upper|1[1]',
        'BB.upper|5', 'BB.upper|5[1]',
        'BB.upper|15', 'BB.upper|15[1]',
        'BB.upper|30', 'BB.upper|30[1]',
        'BB.upper|60', 'BB.upper|60[1]',
        'BB.upper|120', 'BB.upper|120[1]',
        'BB.upper|240', 'BB.upper|240[1]',
        'BB.upper|1W', 'BB.upper|1W[1]',
        'BB.upper|1M', 'BB.upper|1M[1]'
    ).where2(
        And(
            col('is_primary') == True,
            col('typespecs').has('common'),
            col('type') == 'stock',
            col('exchange') == 'NSE',
            col('close').between(20, 10000),
            col('active_symbol') == True,
            col('average_volume_10d_calc|5') > 50000,
            col('Value.Traded|5') > 10000000,

        # Filter for stocks that were in a squeeze on the PREVIOUS candle on any timeframe
        Or(
            And(col('BB.upper[1]') < col('KltChnl.upper[1]'),
                col('BB.lower[1]') > col('KltChnl.lower[1]')),
            And(col('BB.upper|1[1]') < col('KltChnl.upper|1[1]'),
                col('BB.lower|1[1]') > col('KltChnl.lower|1[1]')),
            And(col('BB.upper|5[1]') < col('KltChnl.upper|5[1]'),
                col('BB.lower|5[1]') > col('KltChnl.lower|5[1]')),
            And(col('BB.upper|15[1]') < col('KltChnl.upper|15[1]'),
                col('BB.lower|15[1]') > col('KltChnl.lower|15[1]')),
            And(col('BB.upper|30[1]') < col('KltChnl.upper|30[1]'),
                col('BB.lower|30[1]') > col('KltChnl.lower|30[1]')),
            And(col('BB.upper|60[1]') < col('KltChnl.upper|60[1]'),
                col('BB.lower|60[1]') > col('KltChnl.lower|60[1]')),
            And(col('BB.upper|120[1]') < col('KltChnl.upper|120[1]'),
                col('BB.lower|120[1]') > col('KltChnl.lower|120[1]')),
            And(col('BB.upper|240[1]') < col('KltChnl.upper|240[1]'),
                col('BB.lower|240[1]') > col('KltChnl.lower|240[1]')),
            And(col('BB.upper|1W[1]') < col('KltChnl.upper|1W[1]'),
                col('BB.lower|1W[1]') > col('KltChnl.lower|1W[1]')),
            And(col('BB.upper|1M[1]') < col('KltChnl.upper|1M[1]'),
                col('BB.lower|1M[1]') > col('KltChnl.lower|1M[1]')),
            )
        )

    ).order_by(
        'volume|1', ascending=False
    ).limit(200).set_markets('india').set_property(
        'preset', 'all_stocks'
    ).set_property(
        'symbols', {'query': {'types': ['stock', 'fund', 'dr']}}
    )
    # Execute the query
    _, dfNew = query.get_scanner_data()

    if not dfNew.empty :
        current_time = datetime.now().strftime('%H:%M:%S')
        current_DATE = datetime.now().strftime('%d_%m_%y')

        dfNew.insert(3, 'URL',"")
        dfNew['encodedTicker'] = dfNew['ticker'].apply(urllib.parse.quote)
        dfNew['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol="+dfNew['encodedTicker']+" "
        dfNew.insert(0, 'current_timestamp', current_time)

        # Identify which stocks have a fired squeeze
        processed_df = identify_squeeze_fired_across_timeframes(dfNew)

        # Calculate RVOL and Heatmap Score for visualization
        if 'average_volume_10d_calc|5' in processed_df.columns and 'volume|5' in processed_df.columns:
            # Replace 0s or NaNs in the denominator to avoid division by zero errors
            avg_vol = processed_df['average_volume_10d_calc|5'].replace(0, np.nan)
            processed_df['rvol'] = processed_df['volume|5'] / avg_vol
            processed_df['rvol'] = processed_df['rvol'].fillna(0)  # Fill any NaNs that result from division

            # Calculate Heatmap Score, ensuring TotalSqueezeFiredCount exists
            if 'TotalSqueezeFiredCount' in processed_df.columns:
                processed_df['HeatmapScore'] = (processed_df['rvol'] + 1) * processed_df['TotalSqueezeFiredCount']
                print("✅ RVOL and HeatmapScore columns added.")
            else:
                processed_df['HeatmapScore'] = 0
                print("⚠️ TotalSqueezeFiredCount column not found. HeatmapScore set to 0.")

        else:
            print("⚠️ Could not calculate RVOL or HeatmapScore: Volume columns are missing.")
            processed_df['rvol'] = 0
            processed_df['HeatmapScore'] = 0

        # Filter for stocks that have at least one fired squeeze
        filtered_df = processed_df[processed_df['TotalSqueezeFiredCount'] > 0]

        if not filtered_df.empty :
            # Generate the JSON for the heatmap visualization
            generate_heatmap_json(filtered_df)

            # (Optional) Save the detailed CSV report as well
            append_df_to_csv(filtered_df, 'BBSCAN_FIRED_'+str(current_DATE)+'.csv')

            print(f"=="*30)
            writeHeader=False
            print("--- Filtered Results ---")
            print(filtered_df[['ticker', 'Momentum', 'TotalSqueezeFiredCount', 'rvol', 'HeatmapScore']])

    sleep(120)