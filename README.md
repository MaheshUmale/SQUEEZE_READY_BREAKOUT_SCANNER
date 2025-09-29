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

The scanner operates through a sophisticated, multi-step process to identify and qualify squeeze opportunities.

### 1. Initial Scan & Squeeze Detection
The `BBSqueeze.py` script continuously scans the market using the following core logic:
- **The Squeeze Condition**: It identifies stocks where the **Bollinger Bands (BB)** are inside the **Keltner Channels (KC)**. This indicates a period of low volatility and a potential for a powerful move.
- **Filtering**: It applies a set of baseline filters to ensure data quality, including minimum price, volume, and traded value, while focusing on primary common stocks on the NSE.

### 2. Squeeze Strength Calculation
Once a stock is identified as being in a squeeze, the script calculates its **strength**:
- The strength is determined by the ratio of the Keltner Channel width to the Bollinger Band width (`KC Width / BB Width`).
- Squeezes are categorized as **"STRONG"** (ratio >= 1.5) or **"VERY STRONG"** (ratio >= 2).
- **Only "STRONG" and "VERY STRONG" squeezes are kept for further processing.**

### 3. Event Detection: Formed vs. Fired
The script maintains a history of squeezes in a local database to identify two key events:
- **Newly Formed**: A stock that is in a strong squeeze *now* but was not in the previous scan.
- **Recently Fired**: A stock that was in a strong squeeze in the previous scan but is *not* anymore, indicating the squeeze has "fired" and volatility is expanding.

### 4. Fired Squeeze Analysis
For a "fired" squeeze to be considered a valid breakout signal, it must pass two additional checks:
- **Increased Volatility**: The script calculates volatility using the formula `(BB Upper - SMA20) / ATR`. A fired event is only valid if the current volatility is greater than the volatility recorded when the squeeze first formed.
- **Breakout Direction**: The direction of the breakout is determined by price action, not just momentum indicators:
    - **Bullish Breakout**: `Close > BB.upper` AND `BB.upper > KC.upper`.
    - **Bearish Breakout**: `Close < BB.lower` AND `BB.lower < KC.lower`.

### 5. Dynamic RVOL Calculation
To provide the most accurate measure of volume strength, Relative Volume (RVOL) is calculated dynamically based on the relevant timeframe for each event:
- **For "In Squeeze" stocks**, RVOL is calculated for the highest timeframe the stock is currently in a squeeze on.
- **For "Fired Squeeze" stocks**, RVOL is calculated for the timeframe the squeeze fired on.

### 6. Data Generation & Visualization
The script generates three JSON files with the processed data, which are then fetched by the HTML pages to create the real-time dashboard visualizations.

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