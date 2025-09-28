from time import sleep
from datetime import datetime
from tradingview_screener import Query, col, And
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
    Generates a simple JSON file from the DataFrame for the D3 visualization.
    """
    # Ensure required columns exist
    required_cols = ['ticker', 'URL', 'logo']
    if not all(col in df.columns for col in required_cols):
        print("⚠️ Cannot generate JSON. DataFrame is missing one or more required columns.")
        return

    # Create a simple list of stock data
    heatmap_data = []
    for _, row in df.iterrows():
        heatmap_data.append({
            "name": row['ticker'],
            "url": row['URL'],
            "logo": row['logo']
        })

    # Write the JSON file
    with open(output_path, 'w') as f:
        json.dump(heatmap_data, f, indent=4)
    print(f"✅ Heatmap JSON successfully generated at '{output_path}'.")


while True:
    try:
        query = Query().select(
            'name',
            'logoid',
            'close',
            'volume|5',
            'Value.Traded|5',
            'average_volume_10d_calc|5'
        ).where2(
            And(
                col('is_primary') == True,
                col('typespecs').has('common'),
                col('type') == 'stock',
                col('exchange') == 'NSE',
                col('close').between(20, 10000),
                col('active_symbol') == True,
                col('average_volume_10d_calc|5') > 50000,
                col('Value.Traded|5') > 10000000
            )
        ).order_by(
            'volume|1', ascending=False
        ).limit(200).set_markets('india')

        # Execute the query
        _, dfNew = query.get_scanner_data()

        if dfNew is not None and not dfNew.empty:
            current_time = datetime.now().strftime('%H:%M:%S')
            current_DATE = datetime.now().strftime('%d_%m_%y')

            # Add URL
            dfNew['encodedTicker'] = dfNew['ticker'].apply(urllib.parse.quote)
            dfNew['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + dfNew['encodedTicker']
            dfNew.insert(0, 'current_timestamp', current_time)

            # Construct logo URL from logoid
            if 'logoid' in dfNew.columns:
                dfNew['logo'] = dfNew['logoid'].apply(lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')
            else:
                dfNew['logo'] = ''

            # Generate the JSON for the heatmap visualization
            generate_heatmap_json(dfNew)

            # (Optional) Save the detailed CSV report as well
            append_df_to_csv(dfNew, 'BBSCAN_SIMPLE_'+str(current_DATE)+'.csv')

            print(f"=="*30)
            print("--- Results ---")
            print(dfNew[['ticker', 'logo']])

    except Exception as e:
        print(f"An error occurred: {e}")

    sleep(120)