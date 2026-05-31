import streamlit as st  # streamlit = app layout/buttons/dropdowns/metrics
import requests  # requests = talks to CoinGecko API
import pandas as pd  # pandas = cleans and analyzes tables
import plotly.graph_objects as go  # plotly.graph_objects = creates interactive charts


# This controls the browser tab title and page width
st.set_page_config(
    page_title="CryptoLens",
    layout="wide",
    initial_sidebar_state="expanded"
)

# This sets the background to dark and adds a subtle border to metric cards
st.markdown("""
<style>
.stApp { background-color: #0a0a0a; }
[data-testid="metric-container"] {
    background-color: #111;
    border: 1px solid #222;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# This is the main page title
st.title("CryptoLens")

# This is a short project description under the title
st.caption("crypto analytics dashboard — price, momentum, volatility, drawdown, RSI, volume")


# These are the color constants used across all charts
PLOT_BG = "#0a0a0a"
PAPER_BG = "#0a0a0a"
GRID_COLOR = "#1c1c1c"
TEXT_COLOR = "#e8e8e8"
ACCENT = "#c8f000"
MUTED = "#5a5a5a"


# This applies the dark theme to any plotly figure
def apply_dark_theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR, size=11),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            tickfont=dict(color=MUTED, size=10),
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            tickfont=dict(color=MUTED, size=10),
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID_COLOR,
            font=dict(color=MUTED, size=10),
        ),
        margin=dict(l=8, r=8, t=48, b=8),
        hovermode="x unified",
    )

    # Only apply line styling to line/scatter charts.
    # Bar charts do not support the same line property.
    fig.update_traces(
        line=dict(width=1.4),
        selector=dict(type="scatter")
    )

    return fig


# Left side = what the user sees
# Right side = what CoinGecko needs in the API URL
coins = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana"
}

# Creates a dropdown in the sidebar
coin_name = st.sidebar.selectbox(
    "Choose a coin",
    list(coins.keys())
)

# Fixed options reduce API spam compared to a slider
days = st.sidebar.selectbox(
    "Days of data",
    [7, 30, 90],
    index=1
)

# Example: "Bitcoin" -> "bitcoin"
coin_id = coins[coin_name]


@st.cache_data(ttl=900)
def get_coin_data(coin_id="bitcoin", days=30):
    """
    Pulls historical price and volume data from CoinGecko.
    Uses caching so we do not hit the API too many times.
    """

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    params = {
        "vs_currency": "usd",
        "days": days
    }

    response = requests.get(url, params=params)

    # If CoinGecko gives a bad status code, stop cleanly
    if response.status_code != 200:
        st.error(f"CoinGecko API error: {response.status_code}")
        st.write(response.text)
        st.stop()

    # Try converting response to JSON
    try:
        data = response.json()
    except Exception:
        st.error("CoinGecko returned a response that was not valid JSON.")
        st.write(response.text)
        st.stop()

    # Make sure the expected data exists
    if "prices" not in data or "total_volumes" not in data:
        st.error("CoinGecko response did not include price or volume data.")
        st.write(data)
        st.stop()

    price_df = pd.DataFrame(
        data["prices"],
        columns=["timestamp", "price"]
    )

    volume_df = pd.DataFrame(
        data["total_volumes"],
        columns=["timestamp", "volume"]
    )

    price_df["date"] = pd.to_datetime(
        price_df["timestamp"],
        unit="ms"
    )

    volume_df["date"] = pd.to_datetime(
        volume_df["timestamp"],
        unit="ms"
    )

    df = pd.merge(
        price_df[["date", "price"]],
        volume_df[["date", "volume"]],
        how="left",
        on="date"
    )

    return df


def add_analytics(df):
    """
    Adds financial metrics to the raw price table.

    Input:
    DataFrame with date, price, volume

    Output:
    DataFrame with returns, moving averages, volatility, drawdown, and RSI
    """

    df = df.copy()

    # Percent change from one row to the next
    df["return"] = df["price"].pct_change()

    # Same return, but shown as percent instead of decimal
    df["return_percent"] = df["return"] * 100

    # 24 hour moving average
    df["ma_24h"] = df["price"].rolling(window=24).mean()

    # 7 day moving average
    df["ma_7d"] = df["price"].rolling(window=24 * 7).mean()

    # Rolling volatility
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
    df["rsi"] = 100 - (100 / (1 + df["rs"]))

    return df


def generate_signals(df):
    """
    Creates readable market signals from the latest data row.
    """

    latest = df.iloc[-1]
    signals = []

    if latest["price"] > latest["ma_24h"]:
        signals.append(("Price above 24H moving average", True))
    else:
        signals.append(("Price below 24H moving average", False))

    if latest["rsi"] > 70:
        signals.append(("RSI overbought (>70)", False))
    elif latest["rsi"] < 30:
        signals.append(("RSI oversold (<30)", False))
    else:
        signals.append(("RSI neutral", True))

    if latest["rolling_vol_24h"] > df["rolling_vol_24h"].mean():
        signals.append(("Volatility above average", False))
    else:
        signals.append(("Volatility below average", True))

    if latest["volume"] > df["volume"].mean():
        signals.append(("Volume above average", True))
    else:
        signals.append(("Volume below average", False))

    return signals


def get_market_label(df):
    """
    Creates one overall label for the current market setup.
    """

    latest = df.iloc[-1]

    if latest["price"] > latest["ma_24h"] and latest["rsi"] < 70:
        return "BULLISH", "#c8f000"

    elif latest["price"] < latest["ma_24h"] and latest["rsi"] > 30:
        return "WEAK", "#ff6b35"

    elif latest["rsi"] > 70:
        return "OVERBOUGHT", "#ff4444"

    elif latest["rsi"] < 30:
        return "OVERSOLD", "#00b8ff"

    else:
        return "NEUTRAL", "#5a5a5a"


def plot_price_chart(df, coin_name):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["price"],
        name="Price",
        line=dict(color=ACCENT, width=1.6)
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["ma_24h"],
        name="MA 24H",
        line=dict(color="#00b8ff", width=1.0, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["ma_7d"],
        name="MA 7D",
        line=dict(color="#ff6b35", width=1.0, dash="dash")
    ))

    return apply_dark_theme(fig, f"{coin_name} Price with Moving Averages")


def plot_returns_chart(df, coin_name):
    colors = [ACCENT if v >= 0 else "#ff4444" for v in df["return_percent"]]

    fig = go.Figure(go.Bar(
        x=df["date"],
        y=df["return_percent"],
        marker_color=colors,
        name="Return %",
        marker_line_width=0
    ))

    return apply_dark_theme(fig, f"{coin_name} Returns Over Time")


def plot_volatility_chart(df, coin_name):
    fig = go.Figure(go.Scatter(
        x=df["date"],
        y=df["rolling_vol_24h"],
        fill="tozeroy",
        fillcolor="rgba(200,240,0,0.06)",
        line=dict(color=ACCENT, width=1.4),
        name="24H Volatility"
    ))

    return apply_dark_theme(fig, f"{coin_name} 24H Rolling Volatility")


def plot_drawdown_chart(df, coin_name):
    fig = go.Figure(go.Scatter(
        x=df["date"],
        y=df["drawdown"] * 100,
        fill="tozeroy",
        fillcolor="rgba(255,68,68,0.08)",
        line=dict(color="#ff4444", width=1.4),
        name="Drawdown %"
    ))

    return apply_dark_theme(fig, f"{coin_name} Drawdown")


def plot_rsi_chart(df, coin_name):
    fig = go.Figure()

    fig.add_hrect(
        y0=70,
        y1=100,
        fillcolor="rgba(255,68,68,0.05)",
        line_width=0
    )

    fig.add_hrect(
        y0=0,
        y1=30,
        fillcolor="rgba(0,184,255,0.05)",
        line_width=0
    )

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["rsi"],
        line=dict(color=ACCENT, width=1.4),
        name="RSI"
    ))

    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#ff4444",
        line_width=0.8,
        annotation_text="70",
        annotation_font_color=MUTED,
        annotation_font_size=9
    )

    fig.add_hline(
        y=30,
        line_dash="dot",
        line_color="#00b8ff",
        line_width=0.8,
        annotation_text="30",
        annotation_font_color=MUTED,
        annotation_font_size=9
    )

    fig.update_yaxes(range=[0, 100])

    return apply_dark_theme(fig, f"{coin_name} RSI")


def plot_volume_chart(df, coin_name):
    avg = df["volume"].mean()
    colors = [ACCENT if v >= avg else MUTED for v in df["volume"]]

    fig = go.Figure(go.Bar(
        x=df["date"],
        y=df["volume"],
        marker_color=colors,
        name="Volume",
        marker_line_width=0
    ))

    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color=MUTED,
        line_width=0.8
    )

    return apply_dark_theme(fig, f"{coin_name} Trading Volume")


# Pull raw data from CoinGecko
df = get_coin_data(coin_id, days)

# Add analytics columns
df = add_analytics(df)

# Get the latest row for metric cards
latest = df.iloc[-1]

# Generate signals and market label
signals = generate_signals(df)
market_label, label_color = get_market_label(df)

st.subheader(f"{coin_name} Dashboard")

# Create four metric cards in one row
col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", f"${latest['price']:,.2f}")
col2.metric("RSI", f"{latest['rsi']:.2f}")
col3.metric("Drawdown", f"{latest['drawdown'] * 100:.2f}%")
col4.metric("Market Label", market_label)


# Display signals
st.subheader("Signals")

for label, positive in signals:
    color = "green" if positive else "red"
    st.markdown(f":{color}[- {label}]")


st.plotly_chart(
    plot_price_chart(df, coin_name),
    width="stretch",
    key="price_chart"
)

st.plotly_chart(
    plot_returns_chart(df, coin_name),
    width="stretch",
    key="returns_chart"
)

# Volatility and drawdown side by side since they are related
c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(
        plot_volatility_chart(df, coin_name),
        width="stretch",
        key="volatility_chart"
    )

with c2:
    st.plotly_chart(
        plot_drawdown_chart(df, coin_name),
        width="stretch",
        key="drawdown_chart"
    )

st.plotly_chart(
    plot_rsi_chart(df, coin_name),
    width="stretch",
    key="rsi_chart"
)

st.plotly_chart(
    plot_volume_chart(df, coin_name),
    width="stretch",
    key="volume_chart"
)

# Raw data table
st.subheader("Raw Data")
st.dataframe(df.tail(20))
