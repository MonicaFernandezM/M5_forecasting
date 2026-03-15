import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import math

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

PRIMARY = "#7B61FF"
ACCENT = "#00C2A8"
WARNING = "#FF6B6B"

# --------------------------------------------------
# STYLE
# --------------------------------------------------

st.markdown("""
<style>
.block-container{
padding-top:2rem;
}

[data-testid="metric-container"]{
background:#1E1E2E;
border-radius:10px;
padding:15px;
}
</style>
""", unsafe_allow_html=True)

st.title("Retail Demand Forecasting Dashboard")

st.markdown(
"""
Demand Forecasting and Inventory Optimization to Improve Replenishment Decisions.
"""
)

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

st.sidebar.header("Filter")

store = st.sidebar.selectbox("Store", df["store_id"].unique())
df = df[df["store_id"] == store]

category = st.sidebar.selectbox("Category", sorted(df["cat_id"].unique()))
df = df[df["cat_id"] == category]

department = st.sidebar.selectbox("Department", sorted(df["dept_id"].unique()))
df = df[df["dept_id"] == department]

start_date = st.sidebar.date_input("Start Date", df["date"].min())
end_date = st.sidebar.date_input("End Date", df["date"].max())

df = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

current_inventory = st.sidebar.slider("Estimated Current Inventory", 0, 50, 20)
lead_time = st.sidebar.slider("Lead time", 1, 10, 3)

# --------------------------------------------------
# GLOBAL KPIs
# --------------------------------------------------

st.markdown("### Business Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Sales", f"{df['sales'].sum():,.0f}")
c2.metric("Average Daily Sales", f"{df['sales'].mean():.2f}")
c3.metric("Products", df["item_id"].nunique())
c4.metric("Departments", df["dept_id"].nunique())

# --------------------------------------------------
# CALCULATE REPLENISHMENT TABLE (USED BY MULTIPLE TABS)
# --------------------------------------------------

recommendations = []

for item in df["item_id"].unique():

    item_data = df[df["item_id"] == item]

    demand_mean = item_data["sales"].mean()
    demand_std = item_data["sales"].std()

    safety_stock = 1.65 * demand_std * np.sqrt(lead_time)
    reorder_point = demand_mean * lead_time + safety_stock

    suggested_order = max(0, reorder_point - current_inventory)

    max_reasonable = demand_mean * 10
    suggested_order = min(suggested_order, max_reasonable)

    stockout_risk = demand_mean * lead_time > current_inventory

    recommendations.append([
        item,
        round(demand_mean,2),
        round(safety_stock,2),
        round(reorder_point,2),
        round(suggested_order,2),
        stockout_risk
    ])

rec_df = pd.DataFrame(
    recommendations,
    columns=[
        "Product",
        "Avg Demand",
        "Safety Stock",
        "Reorder Point",
        "Suggested Order",
        "Stockout Risk"
    ]
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Overview",
    "Demand Forecast",
    "Inventory Simulation",
    "Demand Volatility",
    "Replenishment Table",
    "Stockout Risk"
])

# ==================================================
# TAB 1 — DATASET OVERVIEW
# ==================================================

with tab1:

    daily = df.groupby("date")["sales"].sum()
    trend = daily.rolling(30).mean()

    trend_df = pd.DataFrame({
        "date": daily.index,
        "sales": daily.values,
        "trend": trend.values
    })

    col1, col2 = st.columns([2,1])

    with col1:

        fig = px.line(
            trend_df,
            x="date",
            y=["sales","trend"],
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        top_items = (
            df.groupby("item_id")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        st.dataframe(top_items)

# ==================================================
# TAB 2 — DEMAND FORECAST
# ==================================================

with tab2:

    item = st.selectbox(
        "Select Product",
        sorted(df["item_id"].unique())
    )

    item_data = df[df["item_id"] == item].sort_values("date")

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
    forecast_units = math.ceil(forecast)

    weekly = (
        item_data
        .set_index("date")["sales"]
        .resample("W")
        .sum()
        .reset_index()
    )

    # tendencia
    weekly["trend"] = weekly["sales"].rolling(4).mean()

    fig = px.line(
        weekly,
        x="date",
        y="sales",
        template="plotly_dark",
        color_discrete_sequence=[PRIMARY]
    )

    # añadir linea de tendencia
    fig.add_scatter(
        x=weekly["date"],
        y=weekly["trend"],
        mode="lines",
        line=dict(width=4, dash="solid"),
        name="Trend"
    )

    forecast_date = weekly["date"].max() + pd.Timedelta(days=7)

    fig.add_scatter(
        x=[forecast_date],
        y=[forecast_units],
        mode="markers",
        marker=dict(size=12, color=ACCENT),
        name="Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Predicted Demand", forecast_units)
    st.write("Forecast date:", forecast_date.date())

# ==================================================
# TAB 3 — INVENTORY SIMULATION
# ==================================================

with tab3:

    item = st.selectbox("Simulation Product", sorted(df["item_id"].unique()))

    item_data = df[df["item_id"] == item]

    demand_mean = item_data["sales"].mean()
    demand_std = item_data["sales"].std()

    z = 1.65

    safety_stock = math.ceil(z * demand_std * np.sqrt(lead_time))
    reorder_point = math.ceil(demand_mean * lead_time + safety_stock)

    st.write("Safety stock:", round(safety_stock,2))
    st.write("Reorder point:", round(reorder_point,2))

    # inventario inicial
    inventory = current_inventory

    # nivel objetivo de inventario
    target_inventory = 20

    history = []

    for d in range(20):

        demand = np.random.poisson(demand_mean)
        inventory -= demand

        if inventory <= reorder_point:
            order_qty = target_inventory - inventory
            inventory += order_qty

        history.append(inventory)

    sim_df = pd.DataFrame({
        "day": range(20),
        "inventory": history
    })

    fig = px.line(
        sim_df,
        x="day",
        y="inventory",
        template="plotly_dark"
    )

    fig.add_hline(y=reorder_point, line_color=WARNING)
    fig.add_hline(y=safety_stock, line_color=ACCENT)

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 4 — DEMAND VOLATILITY
# ==================================================

with tab4:

    stats = (
        df.groupby("item_id")["sales"]
        .agg(["mean","std"])
        .reset_index()
    )

    stats["cv"] = stats["std"] / stats["mean"]

    st.subheader("Products with Highest Demand Volatility")

    st.dataframe(
        stats.sort_values("cv",ascending=False).head(20)
    )

    fig = px.bar(
        stats.sort_values("cv",ascending=False).head(10),
        x="cv",
        y="item_id",
        orientation="h",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 5 — REPLENISHMENT TABLE
# ==================================================

with tab5:

    st.header("Inventory Replenishment Recommendations")

    lead_time = 3
    service_level = 1.65

    recommendations = []

    items = df["item_id"].unique()[:50]

    for item in items:

        item_data = df[df["item_id"] == item]

        demand_mean = item_data["sales"].mean()
        demand_std = item_data["sales"].std()

        safety_stock = service_level * demand_std * np.sqrt(lead_time)

        reorder_point = demand_mean * lead_time + safety_stock

        order_quantity = demand_mean * lead_time
        order_quantity = min(demand_mean * lead_time * 2, demand_mean * 10)

        recommendations.append([
            item,
            round(demand_mean,2),
            math.ceil(safety_stock),
            math.ceil(reorder_point),
            math.ceil(order_quantity)
        ])

    rec_df = pd.DataFrame(
        recommendations,
        columns=[
            "Product",
            "Avg Daily Demand",
            "Safety Stock",
            "Reorder Point",
            "Suggested Order Qty"
        ]
    )

    st.dataframe(rec_df)

    fig = px.bar(
        rec_df.head(10),
        x="Suggested Order Qty",
        y="Product",
        orientation="h",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 6 — STOCKOUT RISK
# ==================================================

with tab6:

    st.header("Stockout Risk Analysis")

    risk_data = []

    items = df["item_id"].unique()[:50]

    for item in items:

        item_data = df[df["item_id"] == item]

        demand_mean = item_data["sales"].mean()

        demand_leadtime = demand_mean * lead_time

        # inventario estimado (ejemplo: ventas de los últimos días * factor)
        current_inventory = item_data["sales"].tail(7).sum()

        risk = demand_leadtime > current_inventory

        risk_data.append([
            item,
            round(demand_mean,2),
            round(demand_leadtime,2),
            round(current_inventory,2),
            risk
        ])

    risk_df = pd.DataFrame(
        risk_data,
        columns=[
            "Product",
            "Avg Daily Demand",
            "Demand During Lead Time",
            "Current Inventory",
            "Stockout Risk"
        ]
    )

    # Crear columna categórica para el color del gráfico
    risk_df["Risk Status"] = risk_df["Stockout Risk"].apply(
        lambda x: "Stockout Risk" if x else "Inventory OK"
    )

    st.dataframe(risk_df)

    risk_plot = risk_df.sort_values(
        "Demand During Lead Time",
        ascending=False
    ).head(10)

    fig = px.bar(
        risk_plot,
        x="Demand During Lead Time",
        y="Product",
        orientation="h",
        color="Risk Status",
        template="plotly_dark",
        labels={
            "Risk Status": "Inventory Risk"
        },
        color_discrete_map={
            "Stockout Risk": "#FF6B6B",
            "Inventory OK": "#00C2A8"
        }
    )

    st.plotly_chart(fig, use_container_width=True)