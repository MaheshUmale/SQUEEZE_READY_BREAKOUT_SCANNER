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
from tvDatafeed import TvDatafeed,Interval
tv = TvDatafeed()
support_line = defaultdict(float)
resistance_line = defaultdict(float)
import csv


import pandas as pd
import numpy as np


master_combined =  pd.DataFrame()

import pandas as pd
import os

def append_df_to_csv(df, csv_path):
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)



def identify_squeeze_across_timeframes(df):
    """
    Identifies the "squeeze" condition (Keltner Channels inside Bollinger Bands)
    for all available timeframes in the DataFrame, appends new boolean columns,
    and calculates the total number of active squeezes per row.

    The squeeze is defined as:
    KltChnl.upper < BB.upper AND KltChnl.lower > BB.lower

    Args:
        df (pd.DataFrame): The input DataFrame containing the indicator columns.

    Returns:
        pd.DataFrame: The DataFrame with new 'Squeeze|TF' and 'TotalSqueezeCount' columns added.
    """

    # 1. Define all timeframes based on the provided columns (including the base TF)
    # The base timeframe has no '|' suffix.
    timeframes = [''] + ['1', '5', '15', '30', '60', '120', '240', '1W', '1M']
    
    # Check for core columns for demonstration/safety
    if not all(col in df.columns for col in ['KltChnl.upper', 'BB.upper']):
        print("Warning: Missing core columns. Cannot run squeeze logic.")
        return df

    print(f"--- Processing {len(timeframes)} Timeframes ---")

    # List to track the generated squeeze columns for final counting
    squeeze_cols = [] 

    for tf in timeframes:
        # Determine the column suffix
        suffix = f"|{tf}" if tf else ""
        
        # Define the 4 required columns for this timeframe
        kc_upper_col = f'KltChnl.upper{suffix}'
        kc_lower_col = f'KltChnl.lower{suffix}'
        bb_upper_col = f'BB.upper{suffix}'
        bb_lower_col = f'BB.lower{suffix}'
        
        # New column name for the squeeze signal
        squeeze_col = f'Squeeze{suffix}'
        
        # Ensure all columns exist before attempting the comparison
        if all(col in df.columns for col in [kc_upper_col, kc_lower_col, bb_upper_col, bb_lower_col]):
            
            # 2. Apply the Squeeze Logic: (KC_Upper < BB_Upper) AND (KC_Lower > BB_Lower)
            # This is the standard definition of a TTM Squeeze.
            df[squeeze_col] = (
                (df[kc_upper_col] < df[bb_upper_col]) & 
                (df[kc_lower_col] > df[bb_lower_col])
            )
            squeeze_cols.append(squeeze_col) # Track the new column
            print(f"✅ Squeeze column '{squeeze_col}' added.")
            
        else:
            # This handles cases where a column might be missing, which shouldn't happen 
            # if your input column list is consistent, but is good practice.
            print(f"⚠️ Skipping TF '{tf}': One or more required columns are missing.")
            
    # 3. Calculate the total number of active squeezes per row
    if squeeze_cols:
        # Boolean columns are treated as 1 (True) or 0 (False) when summed across axis=1 (rows).
        df['TotalSqueezeCount'] = df[squeeze_cols].sum(axis=1)
        print("✅ TotalSqueezeCount column added.")

    print("--- Processing Complete ---")
    return df




HIGH_VALUE = 50000000.00
VOLUME_MULTIPLIER =  5.0
writeHeader=True
while True : 
  
 
    # tvQuery = Query().set_markets('india') 
    # _,dfNew = (tvQuery
    #  .select('name', 'close', 'volume', 'volume|1','Value.Traded|1','average_volume_10d_calc','average_volume_10d_calc|1')
    # .where(        
    #         col('Value.Traded|1')> HIGH_VALUE 
    #     )
    # # .order_by('relative_volume_intraday|5', ascending=False)
    # .order_by('volume|1', ascending=False).get_scanner_data()) 

    #     query = Query().set_markets('india')
        
        # Select the columns we want to display
    query = Query().select(
        'name',                    # Stock name
        'close',                   # Current price
        'volume|5',                # 1-minute volume
        'Value.Traded|5',          # 1-minute traded value
        'average_volume_10d_calc', # Average volume (10 days)
        'average_volume_10d_calc|5', # Average 1-minute volume (10 days)
        'KltChnl.lower',
        'KltChnl.lower|1',
        'KltChnl.lower|5',
        'KltChnl.lower|15',
        'KltChnl.lower|30',
        'KltChnl.lower|60',
        'KltChnl.lower|120',
        'KltChnl.lower|240',
        'KltChnl.lower|1W',
        'KltChnl.lower|1M',
        'KltChnl.upper',
        'KltChnl.upper|1',
        'KltChnl.upper|5',
        'KltChnl.upper|15',
        'KltChnl.upper|30',
        'KltChnl.upper|60',
        'KltChnl.upper|120',
        'KltChnl.upper|240',
        'KltChnl.upper|1W',
        'KltChnl.upper|1M',
        'BB.lower',
        'BB.lower|1',
        'BB.lower|5',
        'BB.lower|15',
        'BB.lower|30',
        'BB.lower|60',
        'BB.lower|120',
        'BB.lower|240',
        'BB.lower|1W',
        'BB.lower|1M',
        'BB.upper',
        'BB.upper|1',
        'BB.upper|5',
        'BB.upper|15',
        'BB.upper|30',
        'BB.upper|60',
        'BB.upper|120',
        'BB.upper|240',
        'BB.upper|1W',
        'BB.upper|1M'
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
        

        Or(
            And(col('BB.upper') < col('KltChnl.upper'), 
                col('BB.lower') > col('KltChnl.lower')),
            And(col('BB.upper|1') < col('KltChnl.upper|1'), 
                col('BB.lower|1') > col('KltChnl.lower|1')),
            And(col('BB.upper|5') < col('KltChnl.upper|5'), 
                col('BB.lower|5') > col('KltChnl.lower|5')),
            And(col('BB.upper|15') < col('KltChnl.upper|15'), 
                col('BB.lower|15') > col('KltChnl.lower|15')),
            And(col('BB.upper|30') < col('KltChnl.upper|30'), 
                col('BB.lower|30') > col('KltChnl.lower|30')),
            And(col('BB.upper|60') < col('KltChnl.upper|60'), 
                col('BB.lower|60') > col('KltChnl.lower|60')),
            And(col('BB.upper|120') < col('KltChnl.upper|120'), 
                col('BB.lower|120') > col('KltChnl.lower|120')),
            And(col('BB.upper|240') < col('KltChnl.upper|240'), 
                col('BB.lower|240') > col('KltChnl.lower|240')),
            And(col('BB.upper|1W') < col('KltChnl.upper|1W'), 
                col('BB.lower|1W') > col('KltChnl.lower|1W')),
            And(col('BB.upper|1M') < col('KltChnl.upper|1M'), 
                col('BB.lower|1M') > col('KltChnl.lower|1M')), 
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

        # dfNew.insert(2, 'baseURL', "https://in.tradingview.com/chart/N8zfIJVK/?symbol="+str(dfNew['ticker'])+"") 
        # Insert the current time as a new column at the beginning of the DataFrame
        dfNew.insert(3, 'URL',"")  
        dfNew['encodedTicker'] = dfNew['ticker'].apply(urllib.parse.quote)
        dfNew['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol="+dfNew['encodedTicker']+" "

        dfNew.insert(0, 'current_timestamp', current_time)
        # Filter rows where col1 is 3 times col2
        # 
         
        filtered_df =   identify_squeeze_across_timeframes(dfNew )
        
        if not filtered_df.empty : 
            append_df_to_csv(filtered_df, 'BBSCAN_'+str(current_DATE)+'.csv')
            print(f"=="*30)
            # print(filtered_df)
            writeHeader=False 
            print(filtered_df)
        
               
                
    sleep(120)
 