# SQUEEZE_READY_BREAKOUT_SCANNER
TTM SQUEEZE Ready for beak-out on higher time frame
THIS CODE WILL DO THE BACKTESTING OF STOCKS On DAILY,WEEKLY and MONTHLY TIMEFRAME BASED ON ALGORITHM BELOW for NSE INDIA.

The following algorithm details the process for forecasting significant market breakouts by combining volatility compression (representing built-up energy) and volume (representing the necessary fuel), based on the analysis of historical powerful patterns  

# IT WILL USE 
 
from tradingview_screener import Query, col, And, Or
for SCANNING (as shown in BBSqueeze.py file)
 

USe below ALGORIThM Steps to find potential candidate and trigger price 
Check volume levels on breakout of trigger price
after confirming conditions mentioned below enter trade.
follow below algorith detailed steps to enter set stop loss and exit 
Detailed algorithm as below :

***

## Breakout Forecasting Algorithm: Volatility Compression and Volume

This method relies on identifying a state of significant energy buildup (Volatility Compression) followed by a definitive trigger (Breakout Price Level) confirmed by massive inflow (Significant Volume).

### Phase 1: Identifying Potential Energy (Volatility Compression)

**Step 1.1: Select a Higher Time Frame for Context**
To identify the potential for a major, long-lasting breakout, select a **higher time frame** chart (e.g., weekly or monthly). Knowing where the stock is in its volatility cycle on this higher time frame provides crucial context for trading.

**Step 1.2: Apply and Monitor the Volatility Compression Indicator**
Use an indicator designed to measure volatility compression, such as the **TTM Squeeze indicator** or the underlying **Bollinger Band squeeze concept**.

**Step 1.3: Confirm Significant Energy Buildup (The Squeeze)**
Monitor the indicator for the official "squeeze" signal, which indicates that the stock has built up a significant amount of energy:
*   **TTM Squeeze Signal:** Look for the **red dots**, which signal that volatility has compressed to a significant level.
*   **Definition of Squeeze:** This threshold is defined by the Bollinger Band squeezing *inside* the Kelner Channel.
*   **Interpretation:** When the signal appears on the chosen high time frame, it confirms that there is **"plenty of energy in the tank"** for the stock to make a big move. Volatility compression is viewed as "pure potential".

### Phase 2: Defining the Breakout Trigger Level

**Step 2.1: Identify the Consolidation Range**
While the stock is in the squeeze (consolidation/compression), identify the price boundaries of this range on the chart.

**Step 2.2: Define the Breakout Level**
Establish the specific price level that, if exceeded, confirms the stock is moving out of consolidation and initiating the range expansion. This is the level that the stock is beginning to test.

### Phase 3: Confirming the Fuel (Volume) and Trigger

**Step 3.1: Monitor a Lower Time Frame for the Breakout**
Once the high time frame squeeze is confirmed (Step 1.3) and the breakout level is defined (Step 2.2), monitor a **lower time frame** (e.g., the daily chart) for the actual breakout attempt.

**Step 3.2: Confirm the Volume Outlier**
For the breakout to be considered a strong signal, it must be accompanied by **significant volume**, interpreted as the "fuel" needed for range expansion.
*   **Metric:** The required condition is a volume bar that is **greater than two standard deviations above the mean**.
*   **Significance:** This quantifies the volume bar as an "outlier," which is specifically desired for a powerful move.

**Step 3.3: Final Breakout Confirmation**
The powerful breakout signal is confirmed when **all three conditions** are met:
1.  A confirmed squeeze signal exists on the higher time frame (energy).
2.  Price breaks the defined breakout level (trigger).
3.  The breakout is accompanied by a volume bar that closes above the breakout level and is a greater than two standard deviation outlier (fuel).

### Phase 4: Expected Outcome

**Step 4.1: High Risk/Reward Opportunity**
This combined signal generates an **"incredible risk reward opportunity"** for catching powerful moves.

**Step 4.2: Expected Move Duration (Time Frame Dependence)**
The anticipated duration of the resulting move depends on the time frame used for the initial squeeze signal:
*   **Daily Breakouts:** Result in shorter moves, as it takes less time for the stock to work off the volatility expansion.
*   **Monthly Breakouts:** If the monthly pattern triggers on volume (e.g., greater than two standard deviation volume bar on the weekly or daily closing above the level), the stock could be in a price discovery environment for the **next year or two**.
