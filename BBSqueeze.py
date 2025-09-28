from time import sleep
from datetime import datetime
from tradingview_screener import Query, col, And, Or
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

import urllib.parse
import os
import json
import numpy as np

def append_df_to_csv(df, csv_path):
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def generate_heatmap_json(df, output_path='treemap_data.json'):
    """
    Generates a simple, flat JSON array of stock data for the D3 heatmap.
    """
    # Ensure required columns exist for JSON generation
    required_cols = ['ticker', 'HeatmapScore', 'SqueezeCount', 'rvol', 'URL', 'logo']
    for col in required_cols:
        if col not in df.columns:
            # Provide a default value if a column is missing
            df[col] = 0 if 'Score' in col or 'Count' in col or 'rvol' in col else ''

    # Create a flat list of stock data
    heatmap_data = []
    for _, row in df.iterrows():
        heatmap_data.append({
            "name": row['ticker'],
            "value": row['HeatmapScore'],
            "count": row['SqueezeCount'],
            "rvol": row['rvol'],
            "url": row['URL'],
            "logo": row['logo']
        })

    # Write the JSON file
    with open(output_path, 'w') as f:
        json.dump(heatmap_data, f, indent=4)
    print(f"✅ Flat JSON successfully generated at '{output_path}'.")

# Timeframes to check for a squeeze
timeframes = ['', '|1', '|5', '|15', '|30', '|60', '|120', '|240', '|1W', '|1M']

# Construct select columns for all timeframes
select_cols = [
    'name', 'logoid', 'close', 'volume|5', 'Value.Traded|5', 'average_volume_10d_calc|5'
]
for tf in timeframes:
    select_cols.extend([f'KltChnl.lower{tf}', f'KltChnl.upper{tf}', f'BB.lower{tf}', f'BB.upper{tf}'])

while True:
    try:
        # Construct the 'where' condition for current squeezes
        squeeze_conditions = []
        for tf in timeframes:
            squeeze_conditions.append(
                And(
                    col(f'BB.upper{tf}') < col(f'KltChnl.upper{tf}'),
                    col(f'BB.lower{tf}') > col(f'KltChnl.lower{tf}')
                )
            )

        query = Query().select(
            *select_cols
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
                Or(*squeeze_conditions) # Filter for stocks currently in a squeeze on any timeframe
            )
        ).order_by(
            'volume|5', ascending=False
        ).limit(200).set_markets('india')

        # Execute the query
        _, dfNew = query.get_scanner_data()

        if dfNew is not None and not dfNew.empty:
            current_time = datetime.now().strftime('%H:%M:%S')
            current_DATE = datetime.now().strftime('%d_%m_%y')

            # --- Post-processing ---

            # Add URL
            dfNew['encodedTicker'] = dfNew['ticker'].apply(urllib.parse.quote)
            dfNew['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + dfNew['encodedTicker']
            dfNew.insert(0, 'current_timestamp', current_time)

            # Add logo URL
            if 'logoid' in dfNew.columns:
                dfNew['logo'] = dfNew['logoid'].apply(lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')
            else:
                dfNew['logo'] = ''

            # Calculate how many timeframes the stock is in a squeeze
            squeeze_count_cols = []
            for tf in timeframes:
                col_name = f'InSqueeze{tf}'
                dfNew[col_name] = (dfNew[f'BB.upper{tf}'] < dfNew[f'KltChnl.upper{tf}']) & \
                                  (dfNew[f'BB.lower{tf}'] > dfNew[f'KltChnl.lower{tf}'])
                squeeze_count_cols.append(col_name)

            dfNew['SqueezeCount'] = dfNew[squeeze_count_cols].sum(axis=1)

            # Calculate RVOL
            if 'average_volume_10d_calc|5' in dfNew.columns and 'volume|5' in dfNew.columns:
                avg_vol = dfNew['average_volume_10d_calc|5'].replace(0, np.nan)
                dfNew['rvol'] = dfNew['volume|5'] / avg_vol
                dfNew['rvol'] = dfNew['rvol'].fillna(0)
            else:
                dfNew['rvol'] = 0

            # Calculate Heatmap Score
            dfNew['HeatmapScore'] = (dfNew['rvol'] + 1) * dfNew['SqueezeCount']

            # Generate the JSON for the heatmap
            generate_heatmap_json(dfNew)

            # Save detailed report
            append_df_to_csv(dfNew, 'BBSCAN_IN_SQUEEZE_'+str(current_DATE)+'.csv')

            print(f"=="*30)
            print("--- Squeeze Results ---")
            print(dfNew[['ticker', 'SqueezeCount', 'rvol', 'HeatmapScore', 'logo']])

    except Exception as e:
        print(f"An error occurred: {e}")

    sleep(120)