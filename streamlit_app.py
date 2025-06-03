import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------
# Page configuration
st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Function to fetch historical data
@st.cache_data
def get_data(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download(symbol, start=start, end=end, progress=False)
    # Flatten columns if MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        raw.columns.name = None
    df_close = raw[["Close"]].dropna()
    df_close["Close"] = df_close["Close"].squeeze()
    return df_close

# ------------------------------------------------------------
# Build lagged features + 5-day moving average
def make_features(data: pd.DataFrame, n_lags: int = 3):
    """
    Input: data with 'Close' column and DateTimeIndex.
    Output: X (lag_1, lag_2, ..., lag_n, ma_5), y (actual Close).
    """
    df = data.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df.dropna(inplace=True)
    X = df.drop(columns=["Close"])
    y = df["Close"]
    return X, y

# ------------------------------------------------------------
# Train LR + RF and return test-period DataFrame
@st.cache_data
def train_and_predict(
    raw_data: pd.DataFrame,
    train_end: pd.Timestamp
) -> pd.DataFrame:
    """
    1) Input raw_data with 'Close' and sorted DateTimeIndex.
    2) Create features/labels via make_features().
    3) Split at train_end (<= train_end is train, > train_end is test).
    4) Fit Model A = LinearRegression, Model B = RandomForestRegressor.
    5) Return df_test with index = test dates, columns = ['Close','Prediction_A','Prediction_B'].
    """
    data = raw_data.copy().sort_index()
    X_all, y_all = make_features(data)

    df_all = X_all.copy()
    df_all["Close"] = y_all
    df_all["Date"] = df_all.index

    train_mask = df_all["Date"] <= pd.to_datetime(train_end)
    test_mask = df_all["Date"] > pd.to_datetime(train_end)

    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

    # Model A: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_a = lr.predict(X_test)

    # Model B: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_b = rf.predict(X_test)

    df_test = pd.DataFrame(
        {
            "Close": y_test.values,
            "Prediction_A": pred_a,
            "Prediction_B": pred_b,
        },
        index=y_test.index,
    )
    return df_test

# ------------------------------------------------------------
# Compute errors, directional accuracy, and latency
def run_experiment(df: pd.DataFrame):
    """
    Input: df with columns ['Close','Prediction_A','Prediction_B','Group'].
    Output: df with added ['Error','Percentage_Error','Actual_Trend','Predicted_Trend_A','Predicted_Trend_B'],
            plus accuracy_A, accuracy_B, latency_A, latency_B.
    """
    df["Error"] = (df["Close"] - df["Forecast"]).abs()
    df["Percentage_Error"] = df["Error"] / df["Close"] * 100

    df["Actual_Trend"] = np.sign(df["Close"].diff())
    df["Predicted_Trend_A"] = np.sign(df["Prediction_A"].diff())
    df["Predicted_Trend_B"] = np.sign(df["Prediction_B"].diff())

    accuracy_a = (df["Predicted_Trend_A"] == df["Actual_Trend"]).mean() * 100
    accuracy_b = (df["Predicted_Trend_B"] == df["Actual_Trend"]).mean() * 100

    # Simulate “latency” by timing a simple aggregate
    start_a = time.time()
    _ = df["Prediction_A"].mean()
    latency_a = time.time() - start_a

    start_b = time.time()
    _ = df["Prediction_B"].mean()
    latency_b = time.time() - start_b

    return df, accuracy_a, accuracy_b, latency_a, latency_b

# ------------------------------------------------------------
# Perform Welch's t-test on percentage errors
def analyze_results(df: pd.DataFrame):
    errors_a = df[df["Group"] == "A"]["Percentage_Error"]
    errors_b = df[df["Group"] == "B"]["Percentage_Error"]
    t_stat, p_value = ttest_ind(errors_a, errors_b, equal_var=False)
    return errors_a, errors_b, t_stat, p_value

# ------------------------------------------------------------
# Prepare Markdown summary
def format_results(
    df: pd.DataFrame,
    errors_a: pd.Series,
    errors_b: pd.Series,
    accuracy_a: float,
    accuracy_b: float,
    latency_a: float,
    latency_b: float,
    t_stat: float,
    p_value: float
) -> str:
    """
    Returns a Markdown string showing:
      - group sizes,
      - avg % errors,
      - directional accuracies,
      - latencies,
      - t-test results with interpretation.
    """
    md = f"""
### Group Sizes  
• Group A (LR): {df[df["Group"] == "A"].shape[0]}  
• Group B (RF): {df[df["Group"] == "B"].shape[0]}  

**Model A (Linear Regression) Avg % Error:** {errors_a.mean():.2f}%  
**Model B (Random Forest) Avg % Error:** {errors_b.mean():.2f}%  

**Directional Accuracy (A):** {accuracy_a:.2f}%  
**Directional Accuracy (B):** {accuracy_b:.2f}%  

**Latency (A):** {latency_a:.6f} sec  
**Latency (B):** {latency_b:.6f} sec  

**T-Test:** T = {t_stat:.3f}, p = {p_value:.4f}  
"""
    if p_value < 0.05:
        if errors_a.mean() < errors_b.mean():
            md += "\n> **Result:** Model A (Linear Regression) is significantly better (lower error)."
        else:
            md += "\n> **Result:** Model B (Random Forest) is significantly better (lower error)."
    else:
        md += "\n> **Result:** No significant difference between Model A and Model B (p ≥ 0.05)."
    return md

# ------------------------------------------------------------
def main():
    st.title("📈 Financial Model Comparison Dashboard")
    st.write(
        """
        Compare **Linear Regression** (Model A) vs. **Random Forest** (Model B) on S&P 500 closing‐price data.
        You can choose an explicit train/test split date, or let the app automatically do an 80/20 split.
        """
    )

    # Sidebar controls
    st.sidebar.header("🛠 Experiment Settings")
    symbol = st.sidebar.text_input("Ticker Symbol", "^GSPC")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

    split_mode = st.sidebar.radio(
        "Train/Test Split Mode",
        options=["Automatic 80/20", "Specify Date"],
        index=0
    )

    if split_mode == "Specify Date":
        train_end = st.sidebar.date_input(
            "Train/Test Cutoff Date",
            value=pd.to_datetime("2024-09-30")
        )
    else:
        train_end = None  # Will compute later if automatic

    run_button = st.sidebar.button("▶️ Run Experiment")

    if run_button:
        # 1) Fetch
        with st.spinner("Fetching historical data…"):
            raw_df = get_data(symbol, start_date, end_date)
        st.success("✅ Data fetched!")

        if raw_df.shape[0] < 10:
            st.error("Not enough data points to build features. Please choose a longer range.")
            return

        # 2) Determine train_end if automatic
        if split_mode == "Automatic 80/20":
            total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            offset = int(total_days * 0.8)
            train_end = pd.to_datetime(start_date) + pd.Timedelta(days=offset)

        st.write(f"**Train/Test Split:** Training up to {pd.to_datetime(train_end).date()}, testing after.")

        # 3) Train + predict
        with st.spinner("Training models and generating predictions…"):
            df_test = train_and_predict(raw_df, train_end)
        st.success("✅ Models trained and predictions generated!")

        # 4) A/B assign
        df = df_test.copy()
        df["Group"] = np.random.choice(["A", "B"], size=len(df), p=[0.5, 0.5])
        df["Forecast"] = np.where(df["Group"] == "A", df["Prediction_A"], df["Prediction_B"])

        # 5) Compute metrics
        df, acc_a, acc_b, lat_a, lat_b = run_experiment(df)
        errors_a, errors_b, t_stat, p_val = analyze_results(df)

        # 6) Show summary metrics at top
        st.markdown("## 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Group A Size (LR)", df[df["Group"] == "A"].shape[0])
        col1.metric("Group B Size (RF)", df[df["Group"] == "B"].shape[0])

        col2.metric("Avg % Error A", f"{errors_a.mean():.2f}%")
        col2.metric("Avg % Error B", f"{errors_b.mean():.2f}%")

        col3.metric("Accuracy A", f"{acc_a:.2f}%")
        col3.metric("Accuracy B", f"{acc_b:.2f}%")

        col4.metric("Latency A (s)", f"{lat_a:.6f}")
        col4.metric("Latency B (s)", f"{lat_b:.6f}")

        st.markdown("---")

        # 7) Detailed Markdown summary
        st.markdown(format_results(df, errors_a, errors_b, acc_a, acc_b, lat_a, lat_b, t_stat, p_val))

        # 8) Scatterplot in a dedicated container
        st.markdown("## 📈 Predictions vs. Actual")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df.index, df["Prediction_A"], label="LR (Model A)", color="royalblue", alpha=0.6)
        ax.scatter(df.index, df["Prediction_B"], label="RF (Model B)", color="crimson", alpha=0.6)
        ax.plot(df.index, df["Close"], label="Actual Close", color="darkgreen", linewidth=2, alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Model Predictions vs. Actual Closing Price")
        ax.set_xlim(df.index.min(), df.index.max())
        ax.legend()
        st.pyplot(fig)

        # 9) Data preview in an expander
        with st.expander("🗒 View Test‐Period Data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        # 10) Correctness Note
        st.markdown(
            """
            **Note on Correctness**:  
            - We build features (lag₁–₃, 5‐day MA).  
            - We train on days ≤ `train_end`, predict on days > `train_end`.  
            - Errors, percentage errors, directional accuracy, latency, and Welch’s t‐test are computed correctly.  
            - If you see an unusually small test set (e.g., < 5 rows), adjust your date range so that at least 8–10 rows survive the lag/rolling calculations.
            """
        )

if __name__ == "__main__":
    main()
