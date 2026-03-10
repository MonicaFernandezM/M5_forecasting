import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Pronóstico de Demanda Retail y Simulación de Inventario")

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

st.sidebar.header("Filtros")

store = st.sidebar.selectbox(
    "Tienda",
    df["store_id"].unique()
)

df = df[df["store_id"] == store]

category = st.sidebar.selectbox(
    "Categoría",
    sorted(df["cat_id"].unique())
)

df = df[df["cat_id"] == category]

department = st.sidebar.selectbox(
    "Departamento",
    sorted(df["dept_id"].unique())
)

df = df[df["dept_id"] == department]

start_date = st.sidebar.date_input(
    "Fecha Inicial",
    df["date"].min()
)

end_date = st.sidebar.date_input(
    "Fecha Final",
    df["date"].max()
)

df = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Resumen del Dataset",
    "Pronóstico de Demanda",
    "Simulación de Inventario",
    "Productos Críticos"
])

# ==================================================
# TAB 1 — DATASET OVERVIEW
# ==================================================

with tab1:

    st.header("Resumen del Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de registros", len(df))
    col2.metric("Productos", df["item_id"].nunique())
    col3.metric("Tienda", store)
    col4.metric("Ventas medias diarias", round(df["sales"].mean(),2))

    st.subheader("Tendencia de Ventas")

    daily = df.groupby("date")["sales"].sum()
    trend = daily.rolling(30).mean()

    fig, ax = plt.subplots(figsize=(8,3))

    ax.plot(daily.index, daily.values, color="#8073AC", alpha=0.6, label="Ventas Diarias")
    ax.plot(trend.index, trend.values, color="#ff0e26", linewidth=2, label="Tendencia")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Ventas")

    ax.legend()

    st.pyplot(fig)

    st.subheader("Productos con Más Ventas")

    top_items = (
        df.groupby("item_id")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(top_items)

    st.subheader("Mapa de Estacionalidad")

    pivot = df.pivot_table(
        values="sales",
        index="dayofweek",
        columns="month",
        aggfunc="mean"
    )

    days = ["Lun","Mar","Miér","Jue","Vie","Sab","Dom"]
    months = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

    fig, ax = plt.subplots(figsize=(7,4))

    sns.heatmap(pivot, cmap="Purples", ax=ax)

    ax.set_yticklabels(days)
    ax.set_xticklabels(months)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Día de la Semana")

    st.pyplot(fig)

# ==================================================
# TAB 2 — DEMAND FORECAST
# ==================================================

with tab2:

    st.header("Pronóstico de Demanda")

    item = st.selectbox(
        "Seleccionar Producto",
        sorted(df["item_id"].unique())
    )

    item_data = df[df["item_id"] == item].sort_values("date")

    weekly = (
        item_data
        .set_index("date")["sales"]
        .resample("W")
        .sum()
        .reset_index()
    )

    weekly["trend"] = weekly["sales"].rolling(8).mean()

    fig, ax = plt.subplots(figsize=(8,3))

    ax.plot(weekly["date"], weekly["sales"], color="#8073AC", alpha=0.4, label="Ventas Semanales")
    ax.plot(weekly["date"], weekly["trend"], color="#ff0e26", linewidth=2, label="Tendencia")
    
    ax.legend()

    st.pyplot(fig)

    st.subheader("Predicción de Demanda para Mañana")

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

    st.metric("Demanda prevista para mañana", round(forecast,2))

# ==================================================
# TAB 3 — INVENTORY SIMULATION
# ==================================================

with tab3:

    st.header("Simulación de Política de Inventario")

    item = st.selectbox(
        "Seleccionar Producto para Inventario",
        sorted(df["id"].unique())
    )

    item_data = df[df["id"] == item].sort_values("date")

    initial_inventory = st.slider("Inventario inicial",0,100,30)
    lead_time = st.slider("Tiempo de reposición (Lead Time)",1,10,3)
    service_level = st.slider("Nivel de servicio (%)", 90, 99, 95)

    z_values = {
        90: 1.28,
        95: 1.65,
        97: 1.88,
        98: 2.05,
        99: 2.33
    }

    z = z_values[service_level]

    demand_mean = item_data["sales"].mean()
    demand_std = item_data["sales"].std()

    safety_stock = z * demand_std * np.sqrt(lead_time)
    reorder_point = demand_mean * lead_time + safety_stock

    st.write("Demanda media:", round(demand_mean,2))
    st.write("Stock de seguridad:", round(safety_stock,2))
    st.write("Punto de reposición:", round(reorder_point,2))

    # INVENTORY HEALTH INDICATOR
    if initial_inventory < reorder_point:
        st.error("Inventario por debajo del punto de reposición — se recomienda pedir reposición")
    elif initial_inventory < reorder_point * 1.3:
        st.warning("El inventario se está agotando")
    else:
        st.success("Inventario en buen estado")

    # FORECAST RISK
    predicted_demand = model.predict(features)[0]

    if predicted_demand * lead_time > initial_inventory:
        st.error("Alto riesgo de rotura de stock según el pronóstico")

    inventory = initial_inventory
    days = 30
    inventory_history = []

    for d in range(days):
        demand = np.random.poisson(demand_mean)
        inventory -= demand
        if inventory <= reorder_point:
            inventory += 20
        inventory_history.append(inventory)

    fig, ax = plt.subplots(figsize=(8,4))

    # línea inventario
    ax.plot(inventory_history, marker="o", color= "#8073AC" ,label="Nivel de Inventario")

    # líneas de referencia
    ax.axhline(reorder_point, linestyle="--", color="red", label="Punto de reposición")
    ax.axhline(safety_stock, linestyle=":", color="orange", label="Stock de Seguridad")

    # etiquetas
    ax.set_xlabel("Días")
    ax.set_ylabel("Cantidad de Producto")
    ax.set_title("Simulación de Inventario")

    # ticks claros
    ax.set_xticks(range(0,31,5))

    # grid para lectura
    ax.grid(True, alpha=0.3)

    # leyenda
    ax.legend()

    st.pyplot(fig)

# ===================================================
# TAB 4 — CRITICAL PRODUCTS
# ===================================================

with tab4:

    st.header("Productos Críticos (Variabilidad de Demanda)")

    item_stats = (
        df.groupby("id")["sales"]
        .agg(["mean","std"])
        .reset_index()
    )

    item_stats.rename(columns={
        "mean":"avg_demand",
        "std":"demand_std"
    }, inplace=True)

    item_stats["cv"] = item_stats["demand_std"] / item_stats["avg_demand"]

    item_stats["product"] = item_stats["id"].str.split("_CA").str[0]

    top_critical = item_stats.sort_values(
        "cv",
        ascending=False
    ).head(10)

    st.subheader("Productos con Mayor Volatilidad")

    st.dataframe(top_critical)

    fig, ax = plt.subplots(figsize=(7,4))

    ax.barh(
        top_critical["product"],
        top_critical["cv"], 
        color= "#8073AC"
    )

    ax.set_xlabel("Variabilidad de Demanda (CV)")
    ax.set_title("Productos Más Volátiles")

    ax.invert_yaxis()

    st.pyplot(fig)

    # STOCKOUT RISK TABLE

    lead_time = 3

    item_stats["expected_demand_lt"] = item_stats["avg_demand"] * lead_time

    risk_items = item_stats[item_stats["expected_demand_lt"] > item_stats["avg_demand"]]

    st.subheader("Productos con Riesgo de Rotura de Stock")

    st.dataframe(risk_items.head(10))