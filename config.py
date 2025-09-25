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
    "IRB",
]

# Database file name
DB_FILE = "scanner.db"