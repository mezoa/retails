import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Database connection function
def connect_to_db():
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
        st.error(f"Error connecting to the database: {e}")
        return None

# Load data from database
def load_data():
    conn = connect_to_db()
    if conn is None:
        return None

    query = """
    SELECT c.customer_id, c.segment, f.sales, t.year, t.month
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_time t ON f.time_id = t.time_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Customer Segmentation using K-means clustering
def customer_segmentation(df):
    # Aggregate sales and order count by customer
    customer_data = df.groupby('customer_id').agg({
        'sales': 'sum',
        'segment': 'count'  # This gives us the order count
    }).reset_index()
    customer_data.columns = ['customer_id', 'total_sales', 'total_orders']
    
    # Remove any potential duplicates
    customer_data = customer_data.drop_duplicates(subset='customer_id')
    
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(customer_data[['total_sales', 'total_orders']])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(normalized_data)
    
    return customer_data

# Sales Forecasting using Random Forest Regression
def sales_forecasting(df):
    # Aggregate sales by year and month
    monthly_sales = df.groupby(['year', 'month'])['sales'].sum().reset_index()
    monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
    monthly_sales = monthly_sales.sort_values('date')
    
    # Create time-based features
    monthly_sales['month'] = monthly_sales['date'].dt.month
    monthly_sales['year'] = monthly_sales['date'].dt.year
    monthly_sales['day_of_year'] = monthly_sales['date'].dt.dayofyear
    monthly_sales['quarter'] = monthly_sales['date'].dt.quarter
    
    # Create features (X) and target (y)
    features = ['month', 'year', 'day_of_year', 'quarter']
    X = monthly_sales[features]
    y = monthly_sales['sales'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create future dates for forecasting
    last_date = monthly_sales['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='M')
    future_X = pd.DataFrame({
        'month': future_dates.month,
        'year': future_dates.year,
        'day_of_year': future_dates.dayofyear,
        'quarter': future_dates.quarter
    })
    future_pred = model.predict(future_X)
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'date': pd.concat([monthly_sales['date'], pd.Series(future_dates)]),
        'Actual Sales': pd.concat([pd.Series(y), pd.Series([np.nan] * len(future_dates))]),
        'Predicted Sales': pd.concat([pd.Series(y_pred), pd.Series(future_pred)])
    })
    
    return model, plot_data

# Sales Forecasting using Linear Regression
def sales_forecasting_linear_regression(df):
    # Aggregate sales by year and month
    monthly_sales = df.groupby(['year', 'month'])['sales'].sum().reset_index()
    monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
    monthly_sales = monthly_sales.sort_values('date')
    
    # Create time-based features
    monthly_sales['month'] = monthly_sales['date'].dt.month
    monthly_sales['year'] = monthly_sales['date'].dt.year
    monthly_sales['day_of_year'] = monthly_sales['date'].dt.dayofyear
    monthly_sales['quarter'] = monthly_sales['date'].dt.quarter
    
    # Create features (X) and target (y)
    features = ['month', 'year', 'day_of_year', 'quarter']
    X = monthly_sales[features]
    y = monthly_sales['sales'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create future dates for forecasting
    last_date = monthly_sales['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='M')
    future_X = pd.DataFrame({
        'month': future_dates.month,
        'year': future_dates.year,
        'day_of_year': future_dates.dayofyear,
        'quarter': future_dates.quarter
    })
    future_pred = model.predict(future_X)
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'date': pd.concat([monthly_sales['date'], pd.Series(future_dates)]),
        'Actual Sales': pd.concat([pd.Series(y), pd.Series([np.nan] * len(future_dates))]),
        'Predicted Sales': pd.concat([pd.Series(y_pred), pd.Series(future_pred)])
    })
    
    return model, plot_data

# Streamlit app
def main():
    st.title("Retail Data Mining")
    
    # Load data
    df = load_data()
    if df is None or df.empty:
        st.error("Failed to load data or the dataset is empty. Please check your database connection and query.")
        return
    
    st.header("Customer Segmentation")
    try:
        segmented_customers = customer_segmentation(df)
        st.write("Customer segments based on total sales:")
        st.write(segmented_customers)
    except Exception as e:
        st.error(f"Error in customer segmentation: {e}")
    
    st.header("Sales Forecasting - Random Forest")
    try:
        sales_model_rf, forecast_data_rf = sales_forecasting(df)
        st.write("Random Forest Model for Sales Forecasting:")
        st.write("Forecast data:")
        st.write(forecast_data_rf)
        
        # Display feature importances
        feature_importance_rf = pd.DataFrame({
            'feature': ['month', 'year', 'day_of_year', 'quarter'],
            'importance': sales_model_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        st.write("Feature Importances:")
        st.write(feature_importance_rf)
        
    except Exception as e:
        st.error(f"Error in sales forecasting: {e}")
    
    st.header("Sales Forecasting - Linear Regression")
    try:
        sales_model_lr, forecast_data_lr = sales_forecasting_linear_regression(df)
        st.write("Linear Regression Model for Sales Forecasting:")
        st.write("Forecast data:")
        st.write(forecast_data_lr)
        
        # Display coefficients
        coefficients_lr = pd.DataFrame({
            'feature': ['month', 'year', 'day_of_year', 'quarter'],
            'coefficient': sales_model_lr.coef_
        }).sort_values('coefficient', ascending=False)
        st.write("Coefficients:")
        st.write(coefficients_lr)
        
    except Exception as e:
        st.error(f"Error in sales forecasting: {e}")

if __name__ == "__main__":
    main()