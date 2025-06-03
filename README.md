# 📈 Financial Model Comparison Dashboard

## Overview

This project is a Streamlit-based dashboard that compares two machine-learning models—**Linear Regression** (Model A) and **Random Forest** (Model B)—on historical S&P 500 closing-price data. It simulates an A/B-testing framework by randomly assigning each test date to one of the two models, then computes and displays performance metrics, statistical significance, and interactive visualizations.

---

## Features

1. **Data Ingestion**  
   - Downloads historical S&P 500 “Close” prices using `yfinance`.  
   - Drops weekends/holidays automatically.

2. **Feature Engineering**  
   - Creates lagged features (`lag_1`, `lag_2`, `lag_3`).  
   - Computes a 5-day moving average (`ma_5`).

3. **Train/Test Split**  
   - Option 1: Automatic 80/20 split.  
   - Option 2: User-specified cutoff date.

4. **Model Training**  
   - **Model A**: `LinearRegression` on lagged & rolling features.  
   - **Model B**: `RandomForestRegressor` (100 trees).

5. **A/B Assignment & Evaluation**  
   - Randomly assigns each test date to Model A or Model B (50/50).  
   - Computes absolute & percentage error for each assigned forecast.  
   - Calculates directional accuracy (up/down correctness).  
   - Measures simulated “latency” (mean of predicted values as a placeholder).  
   - Runs Welch’s t-test to determine statistical significance of error differences.

6. **Interactive UI**  
   - Sidebar controls for symbol, date range, and split mode.  
   - Metrics row showing group sizes, average % error, directional accuracy, and latency.  
   - Detailed Markdown summary of results (with p-value interpretation).  
   - Wide scatterplot comparing Model A, Model B, and actual closing prices.  
   - Expandable panel for raw test-period data.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone [https://github.com/hypertech12/ForecastingModel/tree/main]
   cd financial-model-comparison-dashboard
<div align="center"> <a href="https://rcforecastingmodel.streamlit.app"> <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="80" alt="Streamlit Logo" /> </a> <p>Click the Streamlit logo to launch the live app</p> </div>
