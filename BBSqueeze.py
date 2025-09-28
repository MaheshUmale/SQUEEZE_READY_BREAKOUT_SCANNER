import os
import urllib.parse
import json # New import for JSON handling
from time import sleep
from datetime import datetime
import pandas as pd
import numpy as np
from tradingview_screener import Query, col, And, Or

import numpy as np

# ... (rest of your imports and functions)

# --- Configuration for DataFrame Display ---
# Set display options for better console output visibility
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200) 

# --- Global Timeframes ---
# Timeframe list: ['' (Base), '1', '5', '15', '30', '60', '120', '240', '1W', '1M']
TIME_FRAMES = [''] + ['1', '5', '15', '30', '60', '120', '240', '1W', '1M']

# --- Helper Functions for TradingView Field Naming ---

def get_select_field_name(indicator_name, tf):
    """
    Returns the field name for data selection (uses pipe '|').
    Example: ('BB.upper', '1') -> 'BB.upper|1'
    Example: ('BB.upper', '') -> 'BB.upper'
    """
    # This format is correct for the .select() function and DataFrame retrieval
    return f'{indicator_name}|{tf}' if tf else indicator_name

# --- File Handling Functions ---

def append_df_to_csv(df, csv_path):
    """
    Appends a DataFrame to a CSV file. Creates the file with a header if it doesn't 
    exist, otherwise appends without the header.
    """
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

# --- Momentum Logic Function ---

def get_momentum_indicator(macd_hist_value):
    """
    Determines momentum indicator (color proxy) based on MACD Histogram value.
    """
    STRENGTH_THRESHOLD = 0.05
    
    if macd_hist_value > STRENGTH_THRESHOLD:
        return 'G↑' # Dark Green (Strong Up)
    elif macd_hist_value > 0:
        return 'Lg' # Light Green (Weak Up)
    elif macd_hist_value < -STRENGTH_THRESHOLD:
        return 'R↓' # Dark Red (Strong Down)
    elif macd_hist_value < 0:
        return 'Lr' # Light Red (Weak Down)
    else:
        return 'N' # Neutral

# --- Treemap Data Generation Function (NEW) ---


def generate_treemap_json(report_df):
    """
    Converts the report DataFrame into a hierarchical JSON structure for the Treemap.
    Grouping is by the Base TF Momentum (Bullish/Bearish/Neutral).
    
    CRITICAL CHANGE: Size is now determined by a combined heatmap score (RVOL * SqueezeCount).
    Color remains based on Momentum, shaded by RVOL.
    
    NOTE: This assumes 'volume|5', 'average_volume_10d_calc|5', and 
    'Momentum|{TIME_FRAMES[0]}' columns are present in report_df.
    """
    
    # --- 1. Calculate RVOL (Relative Volume) for 5 minutes ---
    volume_col = 'volume|5'
    avg_vol_col = 'average_volume_10d_calc|5'
    
    # Calculate RVOL: Safely divide volume by average volume, defaulting to 0 where avg is 0 or NaN
    report_df['RVOL|5'] = np.divide(
        report_df[volume_col],
        report_df[avg_vol_col],
        out=np.zeros_like(report_df[volume_col], dtype=float),
        where=(report_df[avg_vol_col].astype(float) != 0) & (report_df[avg_vol_col].notna())
    ).round(2)
    
    # --- NEW: Combined Heatmap Score (RVOL * SqueezeCount) for box size ---
    # This score is used for the 'value' property in the JSON, which D3 uses for size.
    # We add 1 to RVOL to ensure that a low RVOL doesn't zero out the size if a squeeze exists.
    report_df['HeatmapScore'] = (report_df['RVOL|5'] + 1) * report_df['TotalSqueezeCount']
    
    # --- 2. Define Groups and Momentum ---
    def get_momentum_group(momentum_indicator):
        if momentum_indicator in ['G↑', 'Lg']:
            return 'Bullish Momentum'
        elif momentum_indicator in ['R↓', 'Lr']:
            return 'Bearish Momentum'
        else:
            return 'Neutral / Cross'
    
    base_momentum_col = f'Momentum|{TIME_FRAMES[0] or "Base"}'
    # Ensure the column exists before applying the function
    if base_momentum_col not in report_df.columns:
        print(f"Error: Momentum column '{base_momentum_col}' not found.")
        return {"name": "Squeeze Stocks", "children": []}
        
    report_df['Group'] = report_df[base_momentum_col].apply(get_momentum_group)
    
    # --- 3. Build Hierarchical JSON Data ---
    treemap_data = {
        "name": "Squeeze Stocks",
        "children": []
    }

    # Initialize primary groups
    groups = report_df['Group'].unique()
    for group in groups:
        treemap_data["children"].append({
            "name": group,
            "children": []
        })

    # Add individual stocks as children
    for index, row in report_df.iterrows():
        
        group_name = row['Group']
        
        # Calculate overall Squeeze Intensity by averaging active squeezes
        intensity_cols = [f'SqzIntensity|{tf}' for tf in TIME_FRAMES]
        active_intensities = [row[col] for col in intensity_cols if col in row and row[col] > 0]
        avg_intensity = np.mean(active_intensities) if active_intensities else 0.0
        
        # --- 4. Construct TradingView URL ---
        # The ticker needs to be URL-encoded, as done in the previous version of your script.
        encoded_ticker = urllib.parse.quote(row['ticker'])
        tv_url = f"https://in.tradingview.com/chart/N8zfIJVK/?symbol={encoded_ticker}"

        stock_data = {
            "name": row['ticker'], # Use ticker for the name
            "value": float(row['HeatmapScore']), # NEW: Size based on HeatmapScore (RVOL * SqueezeCount)
            "count": int(row['TotalSqueezeCount']),
            "momentum": row[base_momentum_col], 
            "intensity": avg_intensity,
            "rvol": float(row['RVOL|5']), # RVOL for color shading
            "url": tv_url # URL for the onclick event
        }
        
        # Find the correct parent group and append
        for group_node in treemap_data["children"]:
            if group_node["name"] == group_name:
                group_node["children"].append(stock_data)
                break
                
    # Save the JSON data
    try:
        with open('treemap_data.json', 'w') as f:
            json.dump(treemap_data, f, indent=4)
        print("✅ Treemap data saved to treemap_data.json")
    except IOError as e:
        print(f"❌ Error saving treemap_data.json: {e}")
        
    return treemap_data

# ... (rest of your script)

# --- Squeeze Logic Function (Kept intact) ---

def identify_squeeze_across_timeframes(df):
    """
    Identifies the "squeeze" condition (BB inside KC), calculates Squeeze Intensity,
    determines Momentum Indicator, and counts total active squeezes.
    """

    if df.empty:
        return df

    if not all(col in df.columns for col in [get_select_field_name('KltChnl.upper', ''), get_select_field_name('BB.upper', '')]):
        print("Warning: Missing core columns. Cannot run squeeze logic.")
        return df

    print(f"--- Processing {len(TIME_FRAMES)} Timeframes ---")

    squeeze_cols = [] 
    
    for tf in TIME_FRAMES:
        tf_name = tf or 'Base'
        
        kc_upper_col = get_select_field_name('KltChnl.upper', tf)
        kc_lower_col = get_select_field_name('KltChnl.lower', tf)
        bb_upper_col = get_select_field_name('BB.upper', tf)
        bb_lower_col = get_select_field_name('BB.lower', tf)
        macd_hist_col = get_select_field_name('MACD.hist', tf)
        
        squeeze_bool_col = f'Squeeze|{tf_name}'
        intensity_col = f'SqzIntensity|{tf_name}'
        momentum_ind_col = f'Momentum|{tf_name}'
        
        required_cols = [kc_upper_col, kc_lower_col, bb_upper_col, bb_lower_col, macd_hist_col]
        if all(col in df.columns for col in required_cols):
            
            # --- 1. Identify Squeeze Condition (Boolean) ---
            df[squeeze_bool_col] = (
                (df[bb_upper_col] < df[kc_upper_col]) & 
                (df[bb_lower_col] > df[kc_lower_col])
            )
            squeeze_cols.append(squeeze_bool_col) 
            
            # --- 2. Calculate Squeeze Intensity (Float) ---
            bb_basis = (df[bb_upper_col] + df[bb_lower_col]) / 2
            klt_band_width = bb_basis - df[kc_lower_col]
            bb_band_width = bb_basis - df[bb_lower_col]

            df[intensity_col] = np.divide(
                klt_band_width,
                bb_band_width,
                out=np.zeros_like(klt_band_width, dtype=float),
                where=(bb_band_width != 0) & df[squeeze_bool_col]
            ).round(3) 
            
            # --- 3. Calculate Momentum Indicator ---
            df[momentum_ind_col] = df[macd_hist_col].apply(get_momentum_indicator)

            print(f"✅ Squeeze, Intensity, and Momentum columns added for TF '{tf_name}'")
            
        else:
            print(f"⚠️ Skipping TF '{tf_name}': One or more required columns are missing.")
            
    # --- 4. Calculate Total Squeeze Count ---
    if squeeze_cols:
        df['TotalSqueezeCount'] = df[squeeze_cols].sum(axis=1)
        print("✅ TotalSqueezeCount column added.")

    print("--- Processing Complete ---")
    return df

# --- Main Application Loop ---

while True : 
    
    # Define the base query to select necessary columns across all timeframes
    query = Query().select(
        'name',                    
        'close',                   
        'volume|5',                
        'Value.Traded|5',          # CRITICAL: Used for treemap size
        'average_volume_10d_calc', 
        'average_volume_10d_calc|5', 
        
        # Keltner Lower & Upper
        *[get_select_field_name(f'KltChnl.{band}', tf) 
          for tf in TIME_FRAMES 
          for band in ['lower', 'upper']],
          
        # Bollinger Band Lower & Upper
        *[get_select_field_name(f'BB.{band}', tf) 
          for tf in TIME_FRAMES 
          for band in ['lower', 'upper']],
          
        # MACD Histogram
        *[get_select_field_name('MACD.hist', tf) 
          for tf in TIME_FRAMES]
          
    ).where2(
        And(
            # Primary Filters (Liquidity and Stock Type)
            col('is_primary') == True,
            col('typespecs').has('common'),
            col('type') == 'stock',
            col('exchange') == 'NSE',
            col('close').between(20, 10000),
            col('active_symbol') == True, 
            col('average_volume_10d_calc|5') > 50000, 
            col('Value.Traded|5') > 10000000,
        
            # Squeeze Filter (Using PIPED field names as instructed)
            Or(
                # Base TF ('')
                And(col('BB.upper') < col('KltChnl.upper'), col('BB.lower') > col('KltChnl.lower')),
                # TF '1'
                And(col('BB.upper|1') < col('KltChnl.upper|1'), col('BB.lower|1') > col('KltChnl.lower|1')),
                # TF '5'
                And(col('BB.upper|5') < col('KltChnl.upper|5'), col('BB.lower|5') > col('KltChnl.lower|5')),
                # TF '15'
                And(col('BB.upper|15') < col('KltChnl.upper|15'), col('BB.lower|15') > col('KltChnl.lower|15')),
                # TF '30'
                And(col('BB.upper|30') < col('KltChnl.upper|30'), col('BB.lower|30') > col('KltChnl.lower|30')),
                # TF '60'
                And(col('BB.upper|60') < col('KltChnl.upper|60'), col('BB.lower|60') > col('KltChnl.lower|60')),
                # TF '120'
                And(col('BB.upper|120') < col('KltChnl.upper|120'), col('BB.lower|120') > col('KltChnl.lower|120')),
                # TF '240'
                And(col('BB.upper|240') < col('KltChnl.upper|240'), col('BB.lower|240') > col('KltChnl.lower|240')),
                # TF '1W'
                And(col('BB.upper|1W') < col('KltChnl.upper|1W'), col('BB.lower|1W') > col('KltChnl.lower|1W')),
                # TF '1M'
                And(col('BB.upper|1M') < col('KltChnl.upper|1M'), col('BB.lower|1M') > col('KltChnl.lower|1M')),
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
   
    if not dfNew.empty:
        current_time = datetime.now().strftime('%H:%M:%S')
        current_DATE = datetime.now().strftime('%d_%m_%y')

        # Prepare the DataFrame with metadata
        dfNew.insert(3, 'URL', "")  
        dfNew['encodedTicker'] = dfNew['ticker'].apply(urllib.parse.quote)
        dfNew['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + dfNew['encodedTicker'] + " "
        dfNew.insert(0, 'current_timestamp', current_time)
         
        # Apply the squeeze and intensity/momentum logic
        filtered_df = identify_squeeze_across_timeframes(dfNew)
        
        if not filtered_df.empty: 
            # Filter the final output to only show symbols with at least one active squeeze
            report_df = filtered_df[filtered_df['TotalSqueezeCount'] > 0].copy()

            if not report_df.empty:
                
                # --- Generate Treemap Data (JSON) ---
                generate_treemap_json(report_df)

                print("==" * 65)
                print(f"Squeeze & Momentum Report ({current_time}) - See SqueezeTreemap.html for visual Treemap.")
                print("==" * 65)
                
                # --- Append to CSV ---
                csv_cols = ['current_timestamp', 'name', 'close', 'TotalSqueezeCount']
                append_df_to_csv(report_df[csv_cols], 'BBSCAN_' + str(current_DATE) + '.csv')
                
            else:
                print(f"No active squeezes found at {current_time}.")
        
    # Sleep for 120 seconds (2 minutes) before the next scan
    sleep(120)
