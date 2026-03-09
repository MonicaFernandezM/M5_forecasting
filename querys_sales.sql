-- creación de base de datos 
CREATE DATABASE retail_forecasting;
USE retail_forecasting;

-- 1. Revenue por categoría
SELECT 
cat_id,
ROUND(SUM(sales * sell_price),2) AS total_revenue
FROM df_model_CA1
GROUP BY cat_id
ORDER BY total_revenue DESC;

-- 2. Top productos por ventas

SELECT 
item_id,
SUM(sales) AS total_units_sold
FROM df_model_CA1
GROUP BY item_id
ORDER BY total_units_sold DESC
LIMIT 10;

-- 3.Ventas por día de la semana
SELECT
dayofweek,
ROUND(AVG(sales),2) AS avg_daily_sales
FROM df_model_CA1
GROUP BY dayofweek
ORDER BY dayofweek;

-- 4. Impacto de eventos
SELECT
has_event,
ROUND(AVG(sales),2) AS avg_sales
FROM df_model_CA1
GROUP BY has_event;

-- 5. Volatilidad de productos

SELECT
item_id,
ROUND(cv,2) AS demand_volatility
FROM item_volatility
ORDER BY demand_volatility DESC
LIMIT 10;

-- 6.Tendencia mensual de ventas
SELECT
year,
month,
SUM(sales) AS monthly_sales
FROM sales
GROUP BY year, month
ORDER BY year, month;

-- 7. Precio vs demanda
SELECT
ROUND(sell_price,1) AS price,
ROUND(AVG(sales),2) AS avg_sales
FROM sales
GROUP BY price
ORDER BY price;

-- 8.Productos con más días sin ventas
SELECT
item_id,
SUM(CASE WHEN sales = 0 THEN 1 ELSE 0 END) AS zero_sales_days
FROM sales
GROUP BY item_id
ORDER BY zero_sales_days DESC
LIMIT 10;