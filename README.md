# Real-Time Stock Squeeze Scanner & Heatmap

This project provides a real-time scanner that identifies stocks where a TTM Squeeze has recently "fired" (i.e., the stock has transitioned from a state of low volatility to high volatility) or are currently in a state of low volatility (in a squeeze). The results are visualized as an interactive dashboard in your web browser, which updates automatically.

## Core Components

-   **`BBSqueeze.py`**: The Python backend script that continuously scans the market using the `tradingview_screener` library. It detects squeezes across multiple timeframes, calculates advanced metrics, and generates JSON files for the frontend.
-   **`SqueezeHeatmap.html`**: A single-page web application that visualizes the data. It uses D3.js to create an interactive dashboard showing both "Squeeze Fired" and "In Squeeze" stocks.
-   **`squeeze_history.db`**: An SQLite database used to store the history of squeezed stocks, enabling the script to accurately detect when a squeeze has "fired".
-   **`requirements.txt`**: Lists the necessary Python dependencies.

## How It Works

1.  The `BBSqueeze.py` script runs in a loop, scanning for all stocks that are currently in a squeeze on any timeframe.
2.  It saves the ticker, timeframe, and calculated volatility for each squeeze into the `squeeze_history.db` database.
3.  By comparing the current list of squeezes with the most recent list from the database, it determines which `(ticker, timeframe)` pairs have "fired".
4.  It generates two JSON files: `treemap_data_in_squeeze.json` (for stocks currently in a squeeze) and `treemap_data_fired.json` (for stocks where a squeeze has fired).
5.  The `SqueezeHeatmap.html` page fetches these JSON files every two minutes and updates the dashboard visualization.

## Advanced Metrics

To provide deeper insight, the scanner calculates several advanced metrics:

### Squeeze Strength

For stocks currently in a squeeze, the strength is classified to identify the most compressed conditions.
-   **Formula:** `SQZStrength = KltChnlWidth / BBwidth`
    -   `BBwidth = BB.upper - BB.lower`
    -   `KltChnlWidth = KltChnl.upper - KltChnl.lower`
-   **Categories:**
    -   `SQZStrength >= 2`: VERY STRONG
    -   `SQZStrength >= 1.5`: STRONG
    -   `SQZStrength > 1`: Regular

### Volatility

To prepare for breakout analysis, the script calculates and stores the volatility for each stock in a squeeze. This historical data will be used to confirm breakouts by checking if volatility is expanding when a squeeze fires.
-   **Formula:** `Volatility = ((BB.upper - SMA20) / 2) / ATR`

## Setup and Usage

### 1. Install Dependencies

First, ensure you have Python 3 installed. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Run the Scanner

Open a terminal and run the Python script. It will start scanning and will generate the JSON data files and the `squeeze_history.db` file in the same directory.

```bash
python3 BBSqueeze.py
```

The script will run continuously and update the data every two minutes.

### 3. View the Dashboard

To avoid browser security errors (CORS), you need to serve the files from a local web server. Python has a simple one built-in.

In your terminal, from the project directory, run:

```bash
python3 -m http.server
```

This will start a server, usually on port 8000. Now, open your web browser and navigate to:

**http://localhost:8000/SqueezeHeatmap.html**

The dashboard will load and will automatically refresh every two minutes to show the latest scanner results. You can filter the "In Squeeze" stocks by their strength and expand or collapse sections for a better view.