import streamlit as st #streamlit = app layout/buttons/dropdowns/metrics

import requests #requests = talks to CoinGecko API
 
import pandas as pd #pandas = cleans and analyzes tables

import plotly.express as px  #plotly.express = creates interactive charts



# This controls the browser tab title and page width
st.set_page_config(
    page_title="CryptoLens",
    layout="wide"
)

# This is the main page title
st.title("CryptoLens")

# This is a short project description under the title
st.write(
    "A crypto analytics dashboard for price, momentum, volatility, drawdown, RSI, and volume."
)


# Left side = what the user sees
# Right side = what CoinGecko needs in the API URL
coins = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana"
}



# Creates a dropdown in the sidebar
# User sees Bitcoin/Ethereum/Solana
coin_name = st.sidebar.selectbox(
    "Choose a coin",
    list(coins.keys())
)



days = st.sidebar.slider(
    "Days of data",
    7,
    90,
    30
)


# Example: "Bitcoin" -> "bitcoin"
coin_id = coins[coin_name]




def get_coin_data(coin_id="bitcoin", days=30):
    """
    Pulls historical price and volume data from CoinGecko.

    Input:
    coin_id = CoinGecko coin name, like "bitcoin"
    days = how many days of historical data we want

    Output:
    DataFrame with date, price, and volume
    """

    # This is the CoinGecko endpoint for historical chart data
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    # These are the URL parameters
    # vs_currency means price is shown in USD
    # days means how far back we want data
    params = {
        "vs_currency": "usd",
        "days": days
    }

    # Send request to CoinGecko
    response = requests.get(url, params=params)

    # Convert API response from JSON text into a Python dictionary
    data = response.json()

    # Convert price data into a DataFrame
    # CoinGecko gives prices as [timestamp, price]
    price_df = pd.DataFrame(
        data["prices"],
        columns=["timestamp", "price"]
    )

    # Convert volume data into a DataFrame
    # CoinGecko gives volumes as [timestamp, volume]
    volume_df = pd.DataFrame(
        data["total_volumes"],
        columns=["timestamp", "volume"]
    )

    # Convert machine timestamp into readable datetime
    # unit="ms" means timestamp is in milliseconds
    price_df["date"] = pd.to_datetime(
        price_df["timestamp"],
        unit="ms"
    )

    volume_df["date"] = pd.to_datetime(
        volume_df["timestamp"],
        unit="ms"
    )

    # Merge price and volume together by date
    # Keep only the clean columns we need
    df = pd.merge(
        price_df[["date", "price"]],
        volume_df[["date", "volume"]],
        how="left",
        on="date"
    )

    return df


# Analytics function

def add_analytics(df):
    """
    Adds financial metrics to the raw price table.

    Input:
    DataFrame with date, price, volume

    Output:
    DataFrame with returns, moving averages, volatility, drawdown, and RSI
    """

    # Make a copy so we do not accidentally change the original DataFrame
    df = df.copy()

    # Percent change from one row to the next
    # Since our data is hourly, this is hourly return
    df["return"] = df["price"].pct_change()

    # Same return, but shown as percent instead of decimal
    df["return_percent"] = df["return"] * 100

    # 24 hour moving average
    # Since data is hourly, 24 rows = about 24 hours
    df["ma_24h"] = df["price"].rolling(window=24).mean()

    # 7 day moving average
    # 24 hours * 7 days = 168 rows
    df["ma_7d"] = df["price"].rolling(window=24 * 7).mean()

    # Rolling volatility
    # Standard deviation of returns over the last 24 hours
    df["rolling_vol_24h"] = df["return"].rolling(window=24).std()

    # Running max = highest price seen so far
    df["running_max"] = df["price"].cummax()

    # Drawdown = how far price is below its previous peak
    df["drawdown"] = (df["price"] - df["running_max"]) / df["running_max"]

    # RSI window
    window = 14

    # Price change from previous row
    df["price_change"] = df["price"].diff()

    # Gains are positive price changes
    df["gain"] = df["price_change"].clip(lower=0)

    # Losses are negative price changes, flipped positive
    df["loss"] = -df["price_change"].clip(upper=0)

    # Average gain over RSI window
    df["avg_gain"] = df["gain"].rolling(window=window).mean()

    # Average loss over RSI window
    df["avg_loss"] = df["loss"].rolling(window=window).mean()

    # Relative strength
    df["rs"] = df["avg_gain"] / df["avg_loss"]

    # RSI formula
    # Above 70 often means overbought
    # Below 30 often means oversold
    df["rsi"] = 100 - (100 / (1 + df["rs"]))

    return df


# -------------------------
# Signal functions
# -------------------------

def generate_signals(df):
    """
    Creates readable market signals from the latest data row.
    """

    # Get the most recent row
    latest = df.iloc[-1]

    # Store text signals here
    signals = []

    # Compare current price to short term moving average
    if latest["price"] > latest["ma_24h"]:
        signals.append("Price above 24H moving average")
    else:
        signals.append("Price below 24H moving average")

    # Interpret RSI
    if latest["rsi"] > 70:
        signals.append("RSI overbought")
    elif latest["rsi"] < 30:
        signals.append("RSI oversold")
    else:
        signals.append("RSI neutral")

    # Compare current volatility to average volatility
    if latest["rolling_vol_24h"] > df["rolling_vol_24h"].mean():
        signals.append("Volatility above average")
    else:
        signals.append("Volatility below average")

    # Compare current volume to average volume
    if latest["volume"] > df["volume"].mean():
        signals.append("Volume above average")
    else:
        signals.append("Volume below average")

    return signals


def get_market_label(df):
    """
    Creates one overall label for the current market setup.
    """

    latest = df.iloc[-1]

    if latest["price"] > latest["ma_24h"] and latest["rsi"] < 70:
        return "Bullish / healthy momentum"

    elif latest["price"] < latest["ma_24h"] and latest["rsi"] > 30:
        return "Weak momentum"

    elif latest["rsi"] > 70:
        return "Potentially overbought"

    elif latest["rsi"] < 30:
        return "Potentially oversold"

    else:
        return "Neutral"



# Chart functions


def plot_price_chart(df, coin_name):
    # Price chart with moving averages
    fig = px.line(
        df,
        x="date",
        y=["price", "ma_24h", "ma_7d"],
        title=f"{coin_name} Price with Moving Averages"
    )
    return fig


def plot_returns_chart(df, coin_name):
    # Return chart shows percent move each period
    fig = px.line(
        df,
        x="date",
        y="return_percent",
        title=f"{coin_name} Returns Over Time"
    )
    return fig


def plot_volatility_chart(df, coin_name):
    # Volatility chart shows how jumpy returns are
    fig = px.line(
        df,
        x="date",
        y="rolling_vol_24h",
        title=f"{coin_name} 24H Rolling Volatility"
    )
    return fig


def plot_drawdown_chart(df, coin_name):
    # Drawdown chart shows how far price is below its previous high
    fig = px.line(
        df,
        x="date",
        y="drawdown",
        title=f"{coin_name} Drawdown"
    )
    return fig


def plot_rsi_chart(df, coin_name):
    # RSI chart shows overbought/oversold momentum
    fig = px.line(
        df,
        x="date",
        y="rsi",
        title=f"{coin_name} RSI"
    )

    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")

    return fig


def plot_volume_chart(df, coin_name):
    # Volume chart shows trading activity
    fig = px.line(
        df,
        x="date",
        y="volume",
        title=f"{coin_name} Trading Volume"
    )
    return fig



df = get_coin_data(coin_id, days)

# Add analytics columns
df = add_analytics(df)

# Get the latest row for metric cards
latest = df.iloc[-1]

# Generate signals and market label
signals = generate_signals(df)
market_label = get_market_label(df)

st.subheader(f"{coin_name} Dashboard")


# Create four metric cards in one row
col1, col2, col3, col4 = st.columns(4)




#Show it as dollars, Use commas, Show 2 decimals:


# Latest price
col1.metric(
    "Price",
    f"${latest['price']:,.2f}"
)

# Latest RSI
col2.metric(
    "RSI",
    f"{latest['rsi']:.2f}"
)

# Latest drawdown
col3.metric(
    "Drawdown",
    f"{latest['drawdown'] * 100:.2f}%"
)

# Overall market label
col4.metric(
    "Market Label",
    market_label
)


# Display signals


st.subheader("Signals")


for signal in signals:
    st.write(f"- {signal}") # Go through every item inside the signals list, one at a time, and display each one on the Streamlit app as a bullet point.




st.plotly_chart(
    plot_price_chart(df, coin_name),
    use_container_width=True
)

st.plotly_chart(
    plot_returns_chart(df, coin_name),
    use_container_width=True
)

st.plotly_chart(
    plot_volatility_chart(df, coin_name),
    use_container_width=True
)

st.plotly_chart(
    plot_drawdown_chart(df, coin_name),
    use_container_width=True
)

st.plotly_chart(
    plot_rsi_chart(df, coin_name),
    use_container_width=True
)

st.plotly_chart(
    plot_volume_chart(df, coin_name),
    use_container_width=True
)


# Raw data table
st.subheader("Raw Data")
st.dataframe(df.tail(20))



