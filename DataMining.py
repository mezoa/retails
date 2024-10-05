import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Set up the plot style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a color palette (using blues to match the image)
    palette = sns.color_palette("Blues", as_cmap=True)
    
    # Visualize the clusters
    scatter = ax.scatter(customer_data['total_sales'], 
                         customer_data['total_orders'], 
                         c=customer_data['Cluster'],
                         cmap=palette,
                         alpha=0.7,
                         s=50)
    
    # Customize the plot
    ax.set_xlabel('Total Sales', fontsize=12)
    ax.set_ylabel('Total Orders', fontsize=12)
    ax.set_title('Customer Segments', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster', fontsize=12)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)
    
    return customer_data
# Sales Forecasting using Linear Regression
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
    
    # Create the plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a color palette (using blues to match the customer segmentation)
    palette = sns.color_palette("Blues", n_colors=2)
    
    ax.plot(plot_data['date'], plot_data['Actual Sales'], label='Actual Sales', color=palette[1])
    ax.plot(plot_data['date'], plot_data['Predicted Sales'], label='Predicted Sales', color=palette[0], linestyle='--')
    
    # Customize the plot
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.set_title('Sales Forecast - Random Forest', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis to show dates nicely
    plt.gcf().autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot in Streamlit
    st.pyplot(fig)
    
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
    
    st.header("Sales Forecasting")
    try:
        sales_model, forecast_data = sales_forecasting(df)
        st.write("Random Forest Model for Sales Forecasting:")
        st.write("Forecast data:")
        st.write(forecast_data)
        
        # Display feature importances
        feature_importance = pd.DataFrame({
            'feature': ['month', 'year', 'day_of_year', 'quarter'],
            'importance': sales_model.feature_importances_
        }).sort_values('importance', ascending=False)
        st.write("Feature Importances:")
        st.write(feature_importance)
        
    except Exception as e:
        st.error(f"Error in sales forecasting: {e}")
    
    # ... (rest of the code remains the same)    
    st.header("Insights and Value")
    st.write("""
    1. Customer Segmentation Insights:
       - We've identified three distinct customer segments based on their total sales.
       - This segmentation can help in targeted marketing and personalized customer service.
       - High-value customers (top cluster) should be given priority and special offers to maintain their loyalty.
       - Mid-value customers have potential for growth and could be targeted with upselling strategies.
       - Low-value customers might benefit from retention campaigns or special promotions to increase their engagement.

    2. Sales Forecasting Insights:
       - The linear regression model provides a simple trend of sales over time.
       - This forecast can help in inventory management and resource allocation.
       - Upward trends suggest potential for expansion, while downward trends may indicate a need for new marketing strategies.
       - Seasonal patterns, if any, can inform when to run promotions or adjust staffing levels.

    Value to the Retail Store:
    - Improved customer relationship management through targeted strategies for each segment.
    - Better inventory management and financial planning based on sales forecasts.
    - Data-driven decision making for marketing campaigns and resource allocation.
    - Potential for increased customer satisfaction and loyalty through personalized approaches.
    - Enhanced ability to identify and capitalize on sales trends and patterns.
    """)

if __name__ == "__main__":
    main()