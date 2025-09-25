import pandas as pd
import pandas_ta as ta
from tvDatafeed import TvDatafeed, Interval
from config import STOCKS, TIMEFRANES, DB_FILE
from database import init_db, get_stock_data, save_stock_data, save_backtest_result, get_backtest_results, get_last_date
import datetime

def fetch_data(symbol, timeframe):
    """
    Fetches historical data for a given symbol and timeframe,
    optimizing to fetch only new data since the last run.
    """
    existing_data = get_stock_data(symbol, timeframe.value)
    tv = TvDatafeed()

    if not existing_data.empty:
        last_date = existing_data.index[-1]
        days_since_last_fetch = (datetime.datetime.now() - last_date.to_pydatetime()).days

        print(f"Fetching incremental data for {symbol} ({timeframe.value})...")
        n_bars_to_fetch = days_since_last_fetch + 5
        new_data = tv.get_hist(symbol=symbol, exchange='NSE', interval=timeframe, n_bars=n_bars_to_fetch)

        if new_data is not None and not new_data.empty:
            # Filter out data that already exists
            new_data_to_save = new_data[new_data.index > last_date]
            if not new_data_to_save.empty:
                save_stock_data(symbol, timeframe.value, new_data_to_save)

            # Merge old and new data
            return pd.concat([existing_data, new_data_to_save]).drop_duplicates()

        return existing_data

    else:
        # No data exists, fetch the full history
        print(f"Fetching full history for {symbol} ({timeframe.value})...")
        initial_data = tv.get_hist(symbol=symbol, exchange='NSE', interval=timeframe, n_bars=5000)
        if initial_data is not None and not initial_data.empty:
            save_stock_data(symbol, timeframe.value, initial_data)
            return initial_data

        return pd.DataFrame() # Return empty dataframe if fetch fails

def find_breakouts(symbol):
    """
    Finds breakout opportunities for a given stock based on the
    volatility compression and volume algorithm.
    """
    print(f"Scanning {symbol} for breakouts...")
    # 1. Get data for all timeframes
    daily_data = fetch_data(symbol, Interval.in_daily)
    weekly_data = fetch_data(symbol, Interval.in_weekly)
    monthly_data = fetch_data(symbol, Interval.in_monthly)

    # 2. Calculate TTM Squeeze on higher timeframes
    weekly_data_with_squeeze = pd.concat([weekly_data, weekly_data.ta.squeeze(lazy=True, detailed=True)], axis=1)
    monthly_data_with_squeeze = pd.concat([monthly_data, monthly_data.ta.squeeze(lazy=True, detailed=True)], axis=1)


    # 3. Identify squeeze on higher timeframes
    for high_tf_data in [weekly_data_with_squeeze, monthly_data_with_squeeze]:
        if 'SQZ_ON' not in high_tf_data.columns:
            continue

        # Find periods of squeeze
        squeeze_periods = high_tf_data[high_tf_data['SQZ_ON'] == True]

        for index, row in squeeze_periods.iterrows():
            # 4. Define breakout level (consolidation high)
            start_date = index
            end_date = start_date + pd.DateOffset(months=1) # Look at the next month for breakout

            consolidation_range = high_tf_data.loc[start_date:end_date]
            if consolidation_range.empty:
                continue

            trigger_price = consolidation_range['high'].max()

            # 5. Check for breakout on daily timeframe
            daily_breakout_candidates = daily_data[(daily_data.index > start_date) & (daily_data.index <= end_date)]

            for daily_index, daily_row in daily_breakout_candidates.iterrows():
                if daily_row['close'] > trigger_price:
                    # 6. Confirm with volume
                    volume_mean = daily_data['volume'].rolling(window=20).mean()
                    volume_std = daily_data['volume'].rolling(window=20).std()

                    if daily_row['volume'] > (volume_mean.loc[daily_index] + 2 * volume_std.loc[daily_index]):
                        print(f"BREAKOUT DETECTED for {symbol} on {daily_index.date()} at {daily_row['close']:.2f}")
                        save_backtest_result(symbol, daily_index.date(), daily_row['close'])
                        break # Move to the next squeeze period

def report_results():
    """Queries and prints the backtest results."""
    results = get_backtest_results()
    if not results:
        print("\nNo breakout opportunities found.")
        return

    print("\n--- Backtest Results ---")
    for row in results:
        print(f"Symbol: {row[0]}, Breakout Date: {row[1]}, Trigger Price: {row[2]:.2f}")
    print("----------------------")

if __name__ == "__main__":
    init_db()
    for stock in STOCKS:
        find_breakouts(stock)

    report_results()