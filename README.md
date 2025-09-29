# Real-Time Multi-Timeframe Squeeze Scanner & Dashboard

This project provides a powerful, real-time scanner that identifies stocks in a TTM Squeeze across multiple timeframes. It's designed to help traders spot potential volatility breakouts by tracking when squeezes form and when they fire with increased volatility.

The results are visualized through a responsive, multi-page web dashboard that updates automatically.

## Key Features

-   **Multi-Timeframe Scanning**: Monitors stocks for squeezes on timeframes from 1 minute to 1 month.
-   **In-Squeeze Heatmap**: A comprehensive heatmap view of all stocks currently in a squeeze, grouped by the highest timeframe.
-   **Newly Formed Squeezes**: A dedicated real-time list of squeezes that have just formed in the latest scan, allowing you to catch them early.
-   **Intelligent Fired Squeezes**: A real-time list of squeezes that have not just fired, but have done so with a verifiable *increase in volatility*, indicating a more significant breakout.
-   **Responsive Dashboard**: The user interface is designed to work seamlessly on both desktop and mobile devices.
-   **Detailed Information**: Tooltips and list views provide key data points like momentum, relative volume (RVOL), and volatility changes.

## The Dashboard Views

The application is split into several focused pages:

1.  **Main Heatmap**: The primary dashboard that contains two main sections: "Squeeze Fired" and "In Squeeze". Both sections present data in a compact, time-grouped grid, providing a comprehensive overview of all squeeze activity.
2.  **Compact View**: A dedicated, standalone view of the "In Squeeze" data, presented in the same compact, time-grouped format as the main heatmap for a focused look at stocks currently in a squeeze.
3.  **Newly Formed**: A real-time list showing only the squeezes that have been identified in the most recent scan cycle.
4.  **Recently Fired**: A real-time list showing only the squeezes that have fired *and* have confirmed an increase in volatility since the squeeze began.

## How It Works

1.  The **`BBSqueeze.py`** script runs continuously in the background.
2.  It uses the `tradingview_screener` library to scan for stocks that meet the squeeze criteria (Bollinger Bands inside Keltner Channels).
3.  It maintains a history of squeezes in a local SQLite database (`squeeze_history.db`).
4.  In each cycle, it compares the current list of squeezes with the previous one to identify **newly formed** and **recently fired** squeezes.
5.  For fired squeezes, it calculates the change in volatility to ensure it's a meaningful event.
6.  The script generates three JSON files (`treemap_data_in_squeeze.json`, `treemap_data_formed.json`, `treemap_data_fired.json`) with the latest data.
7.  The HTML pages (`SqueezeHeatmap.html`, `Formed.html`, `Fired.html`) fetch this data and update the visualizations in real-time.

## Setup and Usage

### 1. Install Dependencies

First, ensure you have Python 3 installed. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Run the Scanner Backend

Open a terminal and run the Python script. It's best to run it in the background so it can continue working.

```bash
python3 BBSqueeze.py &
```

The script will start scanning and will generate the JSON data files in the same directory.

### 3. View the Dashboard

To view the web interface, you need to serve the files from a local web server to avoid browser security errors (CORS). Python has a simple one built-in.

In your terminal, from the project directory, run:

```bash
python3 -m http.server
```

This will start a server, usually on port 8000. Now, open your web browser and navigate to:

**http://localhost:8000/SqueezeHeatmap.html**

From there, you can use the navigation links to switch between the different views. The data on the pages will refresh automatically.