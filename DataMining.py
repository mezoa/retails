# DataMining.py
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
import matplotlib.pyplot as plt
import streamlit as st  # Import Streamlit

# Function to connect to PostgreSQL database
def create_connection():
    try:
        conn = psycopg2.connect(
            dbname="Retail",
            user="postgres",
            password="admin",
            host="localhost",
            port="5423"
        )
        return conn
    except Exception as e:
        return None

# Load data from the database
def load_data(query):
    conn = create_connection()
    if conn:
        try:
            data = pd.read_sql(query, conn)
            return data
        finally:
            conn.close()
    return pd.DataFrame()

# Step 1: Customer Segmentation using K-Means Clustering
def customer_segmentation():
    # Query customer data from the database
    query = """
    SELECT dc.customer_id, SUM(fs.sales) as total_sales, COUNT(fs.order_id) as total_orders
    FROM fact_sales fs
    JOIN dim_customer dc ON fs.customer_id = dc.customer_id
    GROUP BY dc.customer_id
    """
    customer_data = load_data(query)

    # Preprocessing: Normalize the data
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data[['total_sales', 'total_orders']])

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

    return customer_data

# Function to forecast sales for additional features
def additional_features_forecast(additional_feature, sales_data, X, dp):
    sales_data['month'] = sales_data.index.to_period('M')
    df_sc = sales_data.groupby(['month'] + additional_feature)['sales'].sum().reset_index()
    df_sc = df_sc.set_index(['month'] + additional_feature).unstack(additional_feature).fillna(0)
    y = df_sc.loc[:, 'sales']

    # Align the indices of X and y
    y = y.reindex(X.index, method='ffill').fillna(0)

    X_train = X.loc[:'2017-12']
    X_test = X.loc['2018-01':]
    y_train = y.loc[:'2017-12']
    y_test = y.loc['2018-01':]

    ada = AdaBoostRegressor(n_estimators=200, estimator=LinearRegression(), loss='exponential', learning_rate=0.0001, random_state=21)
    model = MultiOutputRegressor(ada)
    model.fit(X_train, y_train)

    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index.to_timestamp(), columns=y.columns)
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index.to_timestamp(), columns=y.columns)
    y_forecast = pd.DataFrame(model.predict(dp.out_of_sample(steps=12)), index=dp.out_of_sample(steps=12).index, columns=y.columns)

    fig, axes = plt.subplots(len(y.columns) + 1, 1, figsize=(12, (len(y.columns) + 1) * 2), sharex=True, constrained_layout=True)
    fig.suptitle(f'Sales Forecast for Each {additional_feature[0]}', fontsize=20, color='blue')

    for ax, feature in zip(axes, y.columns):
        ax.plot(y.index.to_timestamp(), y[feature], color="0.5", linestyle='dashed')
        ax.plot(y_fit[feature])
        ax.plot(y_pred[feature])
        ax.plot(y_forecast[feature])
        ax.set_title(f'{feature}, test accuracy: {r2_score(y_test[feature], y_pred[feature]):.2f}', fontsize=10)

    axes[-1].plot(y.index.to_timestamp(), y.sum(axis=1), color="0.5", linestyle='dashed')
    axes[-1].plot(y_fit.sum(axis=1))
    axes[-1].plot(y_pred.sum(axis=1))
    axes[-1].plot(y_forecast.sum(axis=1))
    axes[-1].set_title(f'Sum of all {additional_feature[0]}, test accuracy: {r2_score(y_test.sum(axis=1), y_pred.sum(axis=1)):.2f}', fontsize=12, color='green')
    
    st.pyplot(fig)  # Use Streamlit to display the plot

# Step 2: Sales Forecasting using Linear Regression with AdaBoost
def sales_forecasting():
    # Query sales data by time
    query = """
    SELECT dt.order_date, fs.sales, dp.category, dp.sub_category, dc.segment, dc.region, dc.state, dc.city, fs.ship_mode
    FROM fact_sales fs
    JOIN dim_time dt ON fs.time_id = dt.time_id
    JOIN dim_product dp ON fs.product_id = dp.product_id
    JOIN dim_customer dc ON fs.customer_id = dc.customer_id
    ORDER BY dt.order_date
    """
    sales_data = load_data(query)

    # Preprocessing: Convert order_date to datetime and set as index
    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
    sales_data.set_index('order_date', inplace=True)

    # Remove duplicate entries
    sales_data = sales_data[~sales_data.index.duplicated(keep='first')]

    sales_data = sales_data.asfreq('D')  # Set frequency to daily

    # Deterministic Process for time series
    fourier = CalendarFourier(freq='ME', order=4)  # Use 'ME' instead of 'M'
    dp = DeterministicProcess(
        index=sales_data.index.to_period('M'),  # Convert index to monthly period
        constant=True,
        order=2,
        additional_terms=[fourier],
        seasonal=True,
        drop=True,
    )
    X = dp.in_sample()

    # Forecast for each feature
    features = ['category', 'segment', 'region', 'sub_category', 'state', 'city', 'ship_mode']
    for feature in features:
        additional_features_forecast([feature], sales_data, X, dp)

    return sales_data

# Main function to run data mining tasks and return results
def run_data_mining():
    # Step 1: Run Customer Segmentation
    customer_data = customer_segmentation()
    
    # Step 2: Run Sales Forecasting
    sales_data = sales_forecasting()
    
    return customer_data, sales_data