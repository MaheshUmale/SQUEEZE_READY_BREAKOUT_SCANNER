import sqlite3
import pandas as pd
from config import DB_FILE

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create stock_data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        datetime TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        UNIQUE(symbol, timeframe, datetime)
    )
    """)

    # Create backtest_results table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        breakout_date TEXT NOT NULL,
        trigger_price REAL NOT NULL,
        UNIQUE(symbol, breakout_date)
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def save_stock_data(symbol, timeframe, data):
    """Saves historical stock data to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for index, row in data.iterrows():
        cursor.execute("""
        INSERT OR IGNORE INTO stock_data (symbol, timeframe, datetime, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, timeframe, index.strftime('%Y-%m-%d %H:%M:%S'), row['open'], row['high'], row['low'], row['close'], row['volume']))
    conn.commit()
    conn.close()

def get_stock_data(symbol, timeframe):
    """Retrieves historical stock data from the database and returns a DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    query = f"SELECT datetime, open, high, low, close, volume FROM stock_data WHERE symbol = '{symbol}' AND timeframe = '{timeframe}' ORDER BY datetime"
    df = pd.read_sql_query(query, conn, index_col='datetime', parse_dates=['datetime'])
    conn.close()
    return df

def save_backtest_result(symbol, breakout_date, trigger_price):
    """Saves a breakout signal to the backtest_results table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR IGNORE INTO backtest_results (symbol, breakout_date, trigger_price)
    VALUES (?, ?, ?)
    """, (symbol, breakout_date, trigger_price))
    conn.commit()
    conn.close()

def get_backtest_results():
    """Retrieves all backtest results from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, breakout_date, trigger_price FROM backtest_results")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_last_date(symbol, timeframe):
    """Gets the most recent date for a stock and timeframe in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT MAX(datetime) FROM stock_data
    WHERE symbol = ? AND timeframe = ?
    """, (symbol, timeframe))
    last_date = cursor.fetchone()[0]
    conn.close()
    return last_date

if __name__ == "__main__":
    init_db()