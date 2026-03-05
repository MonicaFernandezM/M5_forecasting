import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Demand Forecasting & Inventory Simulation")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("df_temporal_CA1.csv")
df["date"] = pd.to_datetime(df["date"])

df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

model = joblib.load("demand_forecasting_model.pkl")

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Filters")

# STORE
store = st.sidebar.selectbox(
    "Store",
    df["store_id"].unique()
)

df = df[df["store_id"] == store]

# CATEGORY
category = st.sidebar.selectbox(
    "Category",
    sorted(df["cat_id"].unique())
)

df = df[df["cat_id"] == category]

# DEPARTMENT
department = st.sidebar.selectbox(
    "Department",
    sorted(df["dept_id"].unique())
)

df = df[df["dept_id"] == department]

# DATE FILTER
start_date = st.sidebar.date_input(
    "Start date",
    df["date"].min()
)

end_date = st.sidebar.date_input(
    "End date",
    df["date"].max()
)

df = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Dataset Overview",
    "Demand Forecast",
    "Inventory Simulation"
])

# ==================================================
# TAB 1 — DATASET OVERVIEW
# ==================================================

with tab1:

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Products", df["item_id"].nunique())
    col3.metric("Store", store)
    col4.metric("Avg Daily Sales", round(df["sales"].mean(),2))

    # ------------------------------
    # SALES TREND
    # ------------------------------

    st.subheader("Sales Trend")

    daily = df.groupby("date")["sales"].sum()
    trend = daily.rolling(30).mean()

    fig, ax = plt.subplots(figsize=(8,3))

    ax.plot(daily.index, daily.values, color="#457cb0", alpha=0.6, label="Daily")
    ax.plot(trend.index, trend.values, color="#ff0e26", linewidth=2, label="30-Daily Trend")

    ax.legend()

    st.pyplot(fig)

    # ------------------------------
    # TOP PRODUCTS
    # ------------------------------

    st.subheader("Top Products by Sales")

    top_items = (
        df.groupby("item_id")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(top_items)

    # ------------------------------
    # SEASONALITY HEATMAP
    # ------------------------------

    st.subheader("Seasonality Heatmap")
    pivot = df.pivot_table(
        values="sales",
        index="dayofweek",
        columns="month",
        aggfunc="mean"
    )

    # nombres de días
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(7,4))

    sns.heatmap(pivot, cmap="Blues", ax=ax)

    ax.set_yticklabels(days)
    ax.set_xticklabels(months)

    st.pyplot(fig)

# ==================================================
# TAB 2 — DEMAND FORECAST
# ==================================================

with tab2:

    st.header("Demand Forecast")

    item = st.selectbox(
        "Select Product",
        sorted(df["item_id"].unique())
    )

    # filtrar datos del producto
    item_data = df[df["item_id"] == item].sort_values("date")

    # agrupar ventas por semana
    weekly = (
        item_data
        .set_index("date")["sales"]
        .resample("W")
        .sum()
        .reset_index()
    )

    weekly["trend"] = weekly["sales"].rolling(8).mean()

    fig, ax = plt.subplots(figsize=(8,3))

    ax.plot(weekly["date"], weekly["sales"], color="#457cb0", alpha=0.4, label="Weekly Sales")
    ax.plot(weekly["date"], weekly["trend"], color="#ff0e26", linewidth=2, label="Trend")

    ax.legend()

    st.pyplot(fig)

    # --------------------------
    # NEXT DAY FORECAST
    # --------------------------

    st.subheader("Next Day Forecast")

    latest = item_data.iloc[-1]

    features = [[
        latest["sell_price"],
        latest["is_weekend"],
        latest["has_event"],
        latest["dayofweek"],
        latest["month"],
        latest["year"],
        latest["lag_7"],
        latest["lag_28"],
        latest["rolling_mean_7"],
        latest["rolling_mean_28"]
    ]]

    forecast = model.predict(features)[0]

    st.metric("Predicted Demand Tomorrow", round(forecast,2))

# ==================================================
# TAB 3 — INVENTORY SIMULATION
# ==================================================

with tab3:

    st.header("Inventory Policy Simulation")

    initial_inventory = st.slider(
        "Initial Inventory",
        0,100,30
    )

    lead_time = st.slider(
        "Lead Time (days)",
        1,10,3
    )

    service_level = st.slider(
        "Service Level (z)",
        1.0,2.5,1.65
    )

    demand_mean = df["sales"].mean()
    demand_std = df["sales"].std()

    safety_stock = service_level * demand_std * np.sqrt(lead_time)

    reorder_point = demand_mean * lead_time + safety_stock

    st.write("Safety Stock:", round(safety_stock,2))
    st.write("Reorder Point:", round(reorder_point,2))

    inventory = initial_inventory

    days = 30
    inventory_history = []

    for d in range(days):

        demand = np.random.poisson(demand_mean)

        inventory -= demand

        if inventory <= reorder_point:
            inventory += 20

        inventory_history.append(inventory)

    st.subheader("Inventory Simulation")

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(inventory_history, color="#5a88b2", linewidth=2, label="Inventory")
    ax.axhline(reorder_point, linestyle="--", color="#ff0e26")

    ax.set_xlabel("Day")
    ax.set_ylabel("Inventory")

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)