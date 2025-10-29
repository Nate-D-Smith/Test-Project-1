# --- Imports ---
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Download data (AAPL only) ---
# Note: even with a single ticker, yfinance can return MultiIndex columns depending on version/params.
df = yf.download("AAPL", start="2020-01-01", end="2025-10-01")

# --- Normalize to a 1D 'Close' Series ---
if isinstance(df.columns, pd.MultiIndex):
    # Expect levels like ('Close','AAPL'); select the AAPL close as a 1D Series
    close = df['Close']['AAPL']
else:
    # Regular single-level columns: 'Open','High','Low','Close','Adj Close','Volume'
    close = df['Close']

# Safety: ensure it's 1D
close = pd.Series(close, name="Close")

# --- Build plotting DataFrame with a Date column ---
plot_df = close.reset_index()      # columns: ['Date','Close']

# --- Quick sanity checks (optional) ---
print(plot_df.head())
print(plot_df.dtypes)

# --- Plot with seaborn ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df, x='Date', y='Close')
plt.title("Apple (AAPL) Closing Price Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

