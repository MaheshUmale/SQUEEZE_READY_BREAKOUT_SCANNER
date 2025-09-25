# Configuration for the scanner.
from tvDatafeed import Interval

# Timeframes to analyze
TIMEFRANES = {
    "DAILY": Interval.in_daily,
    "WEEKLY": Interval.in_weekly,
    "MONTHLY": Interval.in_monthly,
}

# List of stock symbols to scan
STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "WIPRO",
    "ASIANPAINT", "HCLTECH", "MARUTI", "BAJFINANCE", "TITAN", "SUNPHARMA",
    "TECHM", "NESTLEIND", "POWERGRID", "ULTRACEMCO", "ADANIENT",
    "TATAMOTORS", "ONGC", "TATASTEEL", "JSWSTEEL", "NTPC", "INDUSINDBK",
    "M&M", "COALINDIA", "BAJAJFINSV", "HINDALCO", "DRREDDY", "GRASIM",
    "DIVISLAB", "BAJAJ-AUTO", "BRITANNIA", "HEROMOTOCO", "ADANIPORTS",
    "CIPLA", "UPL", "SBILIFE", "EICHERMOT", "BPCL", "TATACONSUM",
    "APOLLOHOSP", "SHREECEM", "HDFC"
]

# Database file name
DB_FILE = "scanner.db"