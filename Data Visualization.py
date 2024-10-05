# Data Visualization.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from DataMining import connect_to_db, customer_segmentation, sales_forecasting
import psycopg2

# Set Page Configuration
st.set_page_config(page_title="Superstore Data Analysis", layout="wide")

# Establish connection to PostgreSQL database
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
        st.error(f"Error connecting to the database: {e}")
        return None

# Load data from the database
def load_data(query):
    conn = create_connection()
    if conn:
        try:
            data = pd.read_sql(query, conn)
            print(data.columns)  # Print column names for debugging
            return data
        except Exception as e:
            st.error(f"Error reading from the database: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
        finally:
            conn.close()
    return pd.DataFrame()  # Return empty DataFrame if connection fails

# SQL query to retrieve data
query = """
SELECT fs.*, dp.product_name, dp.category, dp.sub_category, 
       dc.customer_name, dc.segment, 
       dt.order_date, dt.ship_date, dt.year, dt.month, dt.day,
       dc.region, dc.state, dc.city  -- Ensure these columns are included
FROM fact_sales fs
JOIN dim_product dp ON fs.product_id = dp.product_id
JOIN dim_customer dc ON fs.customer_id = dc.customer_id
JOIN dim_time dt ON fs.time_id = dt.time_id
"""
data = load_data(query)

# Debugging: Check the DataFrame's columns
if not data.empty:
    print("DataFrame columns:", data.columns.tolist())  # Output the columns to verify

# Convert date columns to datetime using the correct names from the DataFrame
if 'order_date' in data.columns:
    data['order_date'] = pd.to_datetime(data['order_date'])
if 'ship_date' in data.columns:
    data['ship_date'] = pd.to_datetime(data['ship_date'])


# Sidebar Navigation
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Data Overview", "Sales Trends", 
                                         "Category Analysis", "Product Analysis", 
                                         "Location Analysis", "Data Mining"], 
                                         icons=['house', 'database', 'bar-chart', 
                                                'tags', 'box', 'map', 'search'],
                                         menu_icon="cast", default_index=0)

# Function to filter data based on user input
def filter_data(region, state, city, start_date, end_date):
    # Ensure start_date and end_date are converted to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Applying the filters
    filtered_data = data.copy()
    if region != "All":
        filtered_data = filtered_data[filtered_data['region'] == region]  # Update column name if needed
    if state != "All":
        filtered_data = filtered_data[filtered_data['state'] == state]  # Update column name if needed
    if city != "All":
        filtered_data = filtered_data[filtered_data['city'] == city]  # Update column name if needed
    filtered_data = filtered_data[
        (filtered_data['order_date'] >= start_date) & 
        (filtered_data['order_date'] <= end_date)
    ]
    return filtered_data

# Function to create filter UI components
def create_filters():
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Handle NaT values by setting default dates
    min_order_date = data['order_date'].min() if pd.notna(data['order_date'].min()) else pd.Timestamp('2014-01-01')
    max_order_date = pd.Timestamp('2024-12-31')  # Set the latest date to 2024
    
    with col1:
        region = st.selectbox("Select Region", ["All"] + list(data['region'].unique()))
    with col2:
        state = st.selectbox("Select State", ["All"] + list(data['state'].unique()))
    with col3:
        city = st.selectbox("Pick a City", ["All"] + list(data['city'].unique()))
    with col4:
        start_date = st.date_input("Start Date", min_value=min_order_date, value=min_order_date)
    with col5:
        end_date = st.date_input("End Date", max_value=max_order_date, value=max_order_date)
    return region, state, city, start_date, end_date

# Home Page
if selected == "Home":
    st.title("🏬 Superstore Sales Analysis Dashboard")
    st.write("Welcome to the Superstore Sales Analysis dashboard! Explore insights below.")
    
    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Key Metrics
    total_sales = filtered_data['sales'].sum()
    total_orders = filtered_data['order_id'].nunique()
    total_products = filtered_data['product_id'].nunique()
    unique_customers = filtered_data['customer_id'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales", f"${total_sales:,.2f}")
    with col2:
        st.metric("Total Orders", total_orders)
    with col3:
        st.metric("Unique Products", total_products)
    with col4:
        st.metric("Unique Customers", unique_customers)

    # Overview Charts
    st.subheader("Sales Overview")

    # Sales by Category
    category_sales = filtered_data.groupby('category')['sales'].sum().reset_index()
    fig_category = px.bar(category_sales, x='category', y='sales', title='Sales by Category', labels={'sales': 'Total Sales'}, color='category', color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # Sales by Region
    region_sales = filtered_data.groupby('region')['sales'].sum().reset_index()
    fig_region = px.pie(region_sales, values='sales', names='region', title='Sales Distribution by Region', color='region', color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # Top 10 Selling Products
    top_products = filtered_data.groupby('product_name')['sales'].sum().reset_index().sort_values(by='sales', ascending=False).head(10)
    fig_top_products = go.Figure(data=[go.Bar(x=top_products['product_name'], y=top_products['sales'], marker_color=px.colors.qualitative.Pastel)])
    fig_top_products.update_layout(title='Top 10 Selling Products by Sales', xaxis_title='Product Name', yaxis_title='Total Sales', xaxis_tickangle=-45)
    
    # Display the graphs in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_category, use_container_width=True)
        st.plotly_chart(fig_region, use_container_width=True)
    with col2:
        st.plotly_chart(fig_top_products, use_container_width=True)

# Data Overview Page
if selected == "Data Overview":
    st.title("Data Overview")
    st.write("Here is a snapshot of the dataset.")
    
    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Display data
    st.dataframe(filtered_data)

# Sales Trends Page
if selected == "Sales Trends":
    st.title("Sales Trends")
    
    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Create visuals
    if not filtered_data.empty:
        # Bar graph for sales by category
        category_sales = filtered_data.groupby('category')['sales'].sum().reset_index()
        fig_category = px.bar(category_sales, x='category', y='sales', title='Sales by Category', labels={'sales': 'Total Sales'}, color='category', color_discrete_sequence=px.colors.qualitative.Pastel)

        # Pie graph for sales by region
        region_sales = filtered_data.groupby('region')['sales'].sum().reset_index()
        fig_region = px.pie(region_sales, values='sales', names='region', title='Sales Distribution by Region', color='region', color_discrete_sequence=px.colors.qualitative.Pastel)

        # Line graph for sales over time
        sales_trends = filtered_data.groupby('order_date')['sales'].sum().reset_index()
        fig_trends = px.line(sales_trends, x='order_date', y='sales', title='Sales Over Time', labels={'sales': 'Total Sales'}, color_discrete_sequence=px.colors.qualitative.Pastel)

        # Display the graphs in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_category, use_container_width=True)
            st.plotly_chart(fig_region, use_container_width=True)
        with col2:
            st.plotly_chart(fig_trends, use_container_width=True)

# Category Analysis Page
if selected == "Category Analysis":
    st.title("Category Analysis")
    
    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Create visuals
    if not filtered_data.empty:
        # Segment analysis by category
        category_analysis = filtered_data.groupby('category')['sales'].sum().reset_index()
        fig_category_analysis = px.bar(category_analysis, x='category', y='sales', title='Sales by Category', labels={'sales': 'Total Sales'}, color='category', color_discrete_sequence=px.colors.qualitative.Pastel)

        # Segment analysis by sub-category
        sub_category_analysis = filtered_data.groupby('sub_category')['sales'].sum().reset_index()
        fig_sub_category_analysis = px.bar(sub_category_analysis, x='sub_category', y='sales', title='Sales by Sub-Category', labels={'sales': 'Total Sales'}, color='sub_category', color_discrete_sequence=px.colors.qualitative.Pastel)

        # Display the graphs in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_category_analysis, use_container_width=True)
        with col2:
            st.plotly_chart(fig_sub_category_analysis, use_container_width=True)

# Product Analysis Page
if selected == "Product Analysis":
    st.title("Product Analysis")
    
    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Create visuals
    if not filtered_data.empty:
        # Most popular product in each city
        popular_product = filtered_data.groupby(['city', 'product_name'])['sales'].sum().reset_index()
        popular_product = popular_product.sort_values(['city', 'sales'], ascending=[True, False]).drop_duplicates('city')

        fig_popular_product = px.bar(popular_product, x='city', y='sales', color='product_name', title='Most Popular Product in Each City', labels={'sales': 'Total Sales'}, color_discrete_sequence=px.colors.qualitative.Pastel)

        # Display the graph
        st.plotly_chart(fig_popular_product, use_container_width=True)

# Location Analysis Page
if selected == "Location Analysis":
    st.title("Location Analysis")
    
    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Create visuals
    if not filtered_data.empty:
        # Heatmap for sales by state
        if state == "All":
            heatmap_data = filtered_data.groupby('state')['sales'].sum().reset_index()
            fig_heatmap = px.choropleth(heatmap_data, 
                                        locations='state', 
                                        locationmode='USA-states', 
                                        color='sales', 
                                        scope='usa', 
                                        title='Sales by State',
                                        color_continuous_scale=px.colors.sequential.Plasma)
        else:
            heatmap_data = filtered_data.groupby('city')['sales'].sum().reset_index()
            fig_heatmap = px.choropleth(heatmap_data, 
                                        locations='city', 
                                        locationmode='USA-states', 
                                        color='sales', 
                                        scope='usa', 
                                        title=f'Sales by City in {state}',
                                        color_continuous_scale=px.colors.sequential.Plasma)

        # Display the graph
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Data Mining Insights
if selected == "Data Mining":
    st.title("🛠️ Data Mining Insights")

    # Load data for data mining
    df = load_data(query)
    if df.empty:
        st.error("Failed to load data for data mining.")
    else:
        # Customer Segmentation
        st.subheader("Customer Segmentation (K-Means Clustering)")
        try:
            customer_data = customer_segmentation(df)
            st.write("Customers have been segmented into 3 clusters based on their total sales and total orders.")
            st.dataframe(customer_data)

            # Visualize customer clusters
            fig_customer_clusters = px.scatter(customer_data, x='total_sales', y='total_orders', color='Cluster', title='Customer Segments', color_discrete_sequence=px.colors.qualitative.Pastel)
            
            # Organize the customer clusters graph
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_customer_clusters, use_container_width=True)
        except Exception as e:
            st.error(f"Error in customer segmentation: {e}")

        # Sales Forecasting
        st.subheader("Sales Forecasting (Random Forest)")
        try:
            sales_model, forecast_data = sales_forecasting(df)
            st.write("Historical sales data and the forecast for the next 12 months.")
            st.dataframe(forecast_data)

            # Line plot of historical and future sales
            fig_sales_forecast = px.line(forecast_data, x='date', y=['Actual Sales', 'Predicted Sales'], title='Sales Forecast', labels={'value': 'Sales', 'variable': 'Type'}, color_discrete_sequence=px.colors.qualitative.Pastel)
            
            # Organize the sales forecast graph
            with col2:
                st.plotly_chart(fig_sales_forecast, use_container_width=True)
        except Exception as e:
            st.error(f"Error in sales forecasting: {e}")
# Footer
st.markdown("---")
st.write("Developed by A-Line Business Intelligence Team")