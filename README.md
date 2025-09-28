# Real-Time Stock Squeeze "Fired" Scanner & Heatmap

This project provides a real-time scanner that identifies stocks where a TTM Squeeze has recently "fired" (i.e., the stock has transitioned from a state of low volatility to high volatility). The results are visualized as an interactive heatmap in your web browser, which updates automatically.

## Core Components

-   **`BBSqueeze.py`**: The Python backend script that continuously scans the market using the `tradingview_screener` library. It detects when a squeeze has fired across multiple timeframes, determines the breakout momentum (Bullish or Bearish), and generates a `treemap_data.json` file with the results.
-   **`SqueezeHeatmap.html`**: A single-page web application that visualizes the data from `treemap_data.json`. It uses D3.js to create an interactive heatmap where stocks are grouped by momentum and colored by relative volume (RVOL).
-   **`requirements.txt`**: Lists the necessary Python dependencies.

## How It Works

1.  The `BBSqueeze.py` script runs in a loop, scanning for stocks that were in a squeeze on the previous candle but are no longer in a squeeze on the current candle.
2.  For each "fired" squeeze, it calculates the breakout momentum, relative volume (RVOL), and a "Heatmap Score".
3.  The script then writes this data to a `treemap_data.json` file.
4.  The `SqueezeHeatmap.html` page fetches this JSON file every two minutes and updates the heatmap visualization.

## Setup and Usage


### 1. Install Dependencies

First, ensure you have Python 3 installed. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Run the Scanner

Open a terminal and run the Python script. It will start scanning and will generate the `treemap_data.json` file in the same directory.

```bash
python3 BBSqueeze.py
```

The script will run continuously and update the JSON file every two minutes.

### 3. View the Heatmap

Open the `SqueezeHeatmap.html` file in your web browser (e.g., Chrome, Firefox).

The heatmap will load automatically and will refresh every two minutes to show the latest scanner results.

-   **Green cells**: Stocks with bullish momentum.
-   **Red cells**: Stocks with bearish momentum.
-   **Gray cells**: Neutral momentum.

The intensity of the color indicates the Relative Volume (RVOL), with brighter colors signifying higher RVOL. You can hover over any cell to see detailed information and click on it to open the stock's chart on TradingView.