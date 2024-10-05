import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

# Step 2: Sales Forecasting using Linear Regression
def sales_forecasting():
    # Query sales data by time
    query = """
    SELECT dt.order_date, SUM(fs.sales) as total_sales
    FROM fact_sales fs
    JOIN dim_time dt ON fs.time_id = dt.time_id
    GROUP BY dt.order_date
    ORDER BY dt.order_date
    """
    sales_data = load_data(query)

    # Preprocessing: Convert order_date to ordinal (for regression)
    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
    sales_data['order_date_ordinal'] = sales_data['order_date'].map(pd.Timestamp.toordinal)

    # Linear Regression: Predict future sales
    X = sales_data[['order_date_ordinal']]
    y = sales_data['total_sales']
    model = LinearRegression()
    model.fit(X, y)

    # Predict future sales (next 30 days)
    future_dates = pd.date_range(start=sales_data['order_date'].max(), periods=30)
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal)
    future_sales = model.predict(future_dates_ordinal.values.reshape(-1, 1))

    # Create DataFrame for future sales
    future_sales_df = pd.DataFrame({'date': future_dates, 'predicted_sales': future_sales})
    
    return sales_data, future_sales_df

# Main function to run data mining tasks and return results
def run_data_mining():
    # Step 1: Run Customer Segmentation
    customer_data = customer_segmentation()
    
    # Step 2: Run Sales Forecasting
    sales_data, future_sales = sales_forecasting()
    
    return customer_data, sales_data, future_sales
