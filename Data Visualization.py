# Data Visualization.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from DataMining import connect_to_db, customer_segmentation, sales_forecasting, sales_forecasting_linear_regression
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
# if not data.empty:
#    print("DataFrame columns:", data.columns.tolist())  # Output the columns to verify

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
        st.write("**Sales by Category**: This bar chart shows the total sales for each product category. It helps identify which categories are performing well.")
        
        st.plotly_chart(fig_region, use_container_width=True)
        st.write("**Sales Distribution by Region**: This pie chart illustrates the distribution of sales across different regions. It helps understand regional performance.")
        
    with col2:
        st.plotly_chart(fig_top_products, use_container_width=True)
        st.write("**Top 10 Selling Products**: This bar chart highlights the top 10 products by sales. It helps identify the most popular products.")

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
    st.title("📈 Sales Trends")
    st.write("Analyze the sales trends over time and across different categories and regions.")

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
            st.write("**Sales by Category**: This bar chart shows the total sales for each product category. It helps identify which categories are performing well.")
            
            st.plotly_chart(fig_region, use_container_width=True)
            st.write("**Sales Distribution by Region**: This pie chart illustrates the distribution of sales across different regions. It helps understand regional performance.")
            
        with col2:
            st.plotly_chart(fig_trends, use_container_width=True)
            st.write("**Sales Over Time**: This line chart shows the trend of sales over time. It helps identify seasonal patterns and overall sales growth.")

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
            st.write("**Sales by Category**: This bar chart shows the total sales for each product category. It helps identify which categories are performing well.")
            
        with col2:
            st.plotly_chart(fig_sub_category_analysis, use_container_width=True)
            st.write("**Sales by Sub-Category**: This bar chart shows the total sales for each sub-category. It helps identify which sub-categories are performing well within each category.")

# Product Analysis Page
if selected == "Product Analysis":
    st.title("Product Analysis")
    st.write("Analyze the performance of products across different cities and regions. This section provides insights into the most popular products in each city and the top-selling products overall.")

    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Create visuals
    if not filtered_data.empty:
        # Most popular product in each city
        popular_product = filtered_data.groupby(['city', 'product_name'])['sales'].sum().reset_index()
        popular_product = popular_product.sort_values(['city', 'sales'], ascending=[True, False]).drop_duplicates('city')
        fig_popular_product = px.bar(popular_product, x='city', y='sales', color='product_name', title='Most Popular Product in Each City', labels={'sales': 'Total Sales'}, color_discrete_sequence=px.colors.qualitative.Pastel)

        # Top 10 products by sales
        top_products = filtered_data.groupby('product_name')['sales'].sum().reset_index().sort_values(by='sales', ascending=False).head(10)
        fig_top_products = px.bar(top_products, x='product_name', y='sales', title='Top 10 Products by Sales', labels={'sales': 'Total Sales'}, color='product_name', color_discrete_sequence=px.colors.qualitative.Pastel)

        # Sales by product category
        category_sales = filtered_data.groupby('category')['sales'].sum().reset_index()
        fig_category_sales = px.pie(category_sales, values='sales', names='category', title='Sales Distribution by Category', color='category', color_discrete_sequence=px.colors.qualitative.Pastel)

        # Display the graphs with visual separation
        st.plotly_chart(fig_popular_product, use_container_width=True)
        st.markdown("<h3 style='text-align: left;'>Most Popular Product in Each City</h3>", unsafe_allow_html=True)
        st.write("This bar chart shows the most popular product in each city based on total sales. It helps identify which products are favored in different cities.")
        
        st.markdown("---")  # Horizontal line for separation
        
        st.plotly_chart(fig_top_products, use_container_width=True)
        st.markdown("<h3 style='text-align: left;'>Top 10 Products by Sales</h3>", unsafe_allow_html=True)
        st.write("This bar chart highlights the top 10 products by total sales. It helps identify the best-selling products overall.")
        
        st.markdown("---")  # Horizontal line for separation
        
        st.plotly_chart(fig_category_sales, use_container_width=True)
        st.markdown("<h3 style='text-align: left;'>Sales Distribution by Category</h3>", unsafe_allow_html=True)
        st.write("This pie chart illustrates the distribution of sales across different product categories. It helps understand the contribution of each category to the total sales.")

# Function for state coordinates since current data set doesnt include latitude and longtitude
def add_state_coordinates(df):
    # Dictionary of state abbreviations and their approximate center coordinates
    state_coords = {
        'AL': (32.806671, -86.791130), 'AK': (61.370716, -152.404419), 'AZ': (34.168218, -111.930907),
        'AR': (34.751927, -92.131378), 'CA': (36.778259, -119.417931), 'CO': (39.550051, -105.782067),
        'CT': (41.603221, -73.087749), 'DE': (38.910832, -75.527670), 'FL': (27.994402, -81.760254),
        'GA': (32.157435, -82.907123), 'HI': (19.896766, -155.582782), 'ID': (44.068202, -114.742041),
        'IL': (40.633125, -89.398529), 'IN': (40.551217, -85.602364), 'IA': (41.878003, -93.097702),
        'KS': (39.011902, -98.484246), 'KY': (37.839333, -84.270018), 'LA': (30.984298, -91.962333),
        'ME': (45.253783, -69.445469), 'MD': (39.045755, -76.641271), 'MA': (42.407211, -71.382437),
        'MI': (44.314844, -85.602364), 'MN': (46.729553, -94.685900), 'MS': (32.354668, -89.398528),
        'MO': (37.964253, -91.831833), 'MT': (46.879682, -110.362566), 'NE': (41.492537, -99.901813),
        'NV': (38.802610, -116.419389), 'NH': (43.193852, -71.572395), 'NJ': (40.058324, -74.405661),
        'NM': (34.519940, -105.870090), 'NY': (43.299428, -74.217933), 'NC': (35.759573, -79.019300),
        'ND': (47.551493, -101.002012), 'OH': (40.417287, -82.907123), 'OK': (35.007752, -97.092877),
        'OR': (43.804133, -120.554201), 'PA': (41.203322, -77.194525), 'RI': (41.580095, -71.477429),
        'SC': (33.836081, -81.163725), 'SD': (43.969515, -99.901813), 'TN': (35.517491, -86.580447),
        'TX': (31.968599, -99.901813), 'UT': (39.320980, -111.093731), 'VT': (44.558803, -72.577841),
        'VA': (37.431573, -78.656894), 'WA': (47.751074, -120.740139), 'WV': (38.597626, -80.454903),
        'WI': (43.784440, -88.787868), 'WY': (43.075968, -107.290284)
    }

    # Create a mapping of full state names to coordinates
    state_name_to_coords = {state: coords for state, coords in zip(df['state'].unique(), [state_coords.get(state[:2].upper(), (0, 0)) for state in df['state'].unique()])}
    
    # Add latitude and longitude columns
    df['latitude'] = df['state'].map(lambda x: state_name_to_coords.get(x, (0, 0))[0])
    df['longitude'] = df['state'].map(lambda x: state_name_to_coords.get(x, (0, 0))[1])
    
    return df

# Location Analysis Page
if selected == "Location Analysis":
    st.title("Location Analysis")
    st.write("Analyze the sales performance across different states and cities. This section provides insights into the sales distribution by state and city, as well as a geographical representation of sales.")

    # Filters
    region, state, city, start_date, end_date = create_filters()
    filtered_data = filter_data(region, state, city, start_date, end_date)

    # Add coordinates to the filtered data
    filtered_data = add_state_coordinates(filtered_data)

    # Create visuals
    if not filtered_data.empty:
        # First row: State and City bar charts
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for sales by state
            state_sales = filtered_data.groupby('state')['sales'].sum().reset_index()
            state_sales = state_sales.sort_values('sales', ascending=False)
            fig_state_bar = px.bar(state_sales, 
                                   x='state', 
                                   y='sales', 
                                   title='Sales by State',
                                   labels={'sales': 'Total Sales', 'state': 'State'},
                                   color='sales',
                                   color_continuous_scale=px.colors.sequential.Plasma)
            fig_state_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_state_bar, use_container_width=True)
            st.markdown("<h3 style='text-align: left;'>Sales by State</h3>", unsafe_allow_html=True)
            st.write("This bar chart shows the total sales for each state. It helps identify which states are generating the most revenue.")

        with col2:
            # Bar chart for sales by city
            if state != "All":
                city_sales = filtered_data[filtered_data['state'] == state].groupby('city')['sales'].sum().reset_index()
            else:
                city_sales = filtered_data.groupby('city')['sales'].sum().reset_index()
            city_sales = city_sales.sort_values('sales', ascending=False).head(10)  # Top 10 cities
            fig_city_bar = px.bar(city_sales, 
                                  x='city', 
                                  y='sales', 
                                  title=f'Top 10 Cities by Sales',
                                  labels={'sales': 'Total Sales', 'city': 'City'},
                                  color='sales',
                                  color_continuous_scale=px.colors.sequential.Plasma)
            fig_city_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_city_bar, use_container_width=True)
            st.markdown("<h3 style='text-align: left;'>Top 10 Cities by Sales</h3>", unsafe_allow_html=True)
            st.write("This bar chart highlights the top 10 cities by total sales. It helps identify the cities with the highest sales performance.")

        st.markdown("---")  # Horizontal line for separation

        # Second row: Map
        # Scatter plot on map
        location_data = filtered_data.groupby(['state', 'latitude', 'longitude'])['sales'].sum().reset_index()
        
        fig_map = px.scatter_geo(location_data,
                                 lat='latitude',
                                 lon='longitude',
                                 size='sales',
                                 hover_name='state',
                                 scope='usa',
                                 title='Sales Distribution on Map',
                                 size_max=50,
                                 color='sales',
                                 color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("<h3 style='text-align: left;'>Sales Distribution on Map</h3>", unsafe_allow_html=True)
        st.write("This scatter plot on the map shows the distribution of sales across different states. The size of the bubbles represents the total sales in each state, providing a geographical overview of sales performance.")

    else:
        st.warning("No data available for the selected filters.")
        
# Data Mining Insights
if selected == "Data Mining":
    st.title("🛠️ Data Mining Insights")

    # Dropdown for selecting the data mining technique
    technique = st.selectbox("Select a technique:", ["Customer Segmentation", "Sales Forecasting (Random Forest)", "Sales Forecasting (Linear Regression)"])

    # Load data for data mining
    df = load_data(query)
    if df.empty:
        st.error("Failed to load data for data mining.")
    else:
        if technique == "Customer Segmentation":
            st.subheader("Customer Segmentation (K-Means Clustering)")
            try:
                customer_data = customer_segmentation(df)
                st.write("Customers have been segmented into 3 clusters based on their total sales and total orders.")
                
                # Visualize customer clusters
                fig_customer_clusters = px.scatter(customer_data, x='total_sales', y='total_orders', color='Cluster', title='Customer Segments', color_discrete_sequence=px.colors.qualitative.Pastel)
                
                # Display the customer clusters graph and table
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_customer_clusters, use_container_width=True)
                with col2:
                    st.dataframe(customer_data, height=400)
                
                # Explanation
                st.markdown("""
                **Customer Segmentation Insights:**
                - The scatter plot shows the distribution of customers across three clusters based on their total sales and total orders.
                - Each point represents a customer, with the x-axis showing total sales and the y-axis showing total orders.
                - The different colors represent different clusters, helping to identify groups of customers with similar purchasing behaviors.
                - The table on the right provides detailed information about each customer, including their cluster assignment.
                """)
            except Exception as e:
                st.error(f"Error in customer segmentation: {e}")

        elif technique == "Sales Forecasting (Random Forest)":
            st.subheader("Sales Forecasting (Random Forest)")
            try:
                sales_model_rf, forecast_data_rf = sales_forecasting(df)
                st.write("Historical sales data and the forecast for the next 12 months using Random Forest.")
                
                # Line plot of historical and future sales
                fig_sales_forecast_rf = px.line(forecast_data_rf, x='date', y=['Actual Sales', 'Predicted Sales'], title='Sales Forecast (Random Forest)', labels={'value': 'Sales', 'variable': 'Type'}, color_discrete_sequence=px.colors.qualitative.Pastel)
                
                # Display the sales forecast graph and table
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(forecast_data_rf, height=400)
                with col2:
                    st.plotly_chart(fig_sales_forecast_rf, use_container_width=True)
                
                # Explanation
                st.markdown("""
                **Sales Forecasting (Random Forest) Insights:**
                - The line plot shows historical sales data (Actual Sales) and the forecasted sales for the next 12 months (Predicted Sales) using the Random Forest model.
                - The x-axis represents the date, while the y-axis represents the sales amount.
                - The table on the left provides detailed forecast data, including both actual and predicted sales values.
                - This forecast helps in understanding future sales trends and making informed business decisions.
                """)
            except Exception as e:
                st.error(f"Error in sales forecasting: {e}")

        elif technique == "Sales Forecasting (Linear Regression)":
            st.subheader("Sales Forecasting (Linear Regression)")
            try:
                sales_model_lr, forecast_data_lr = sales_forecasting_linear_regression(df)
                st.write("Historical sales data and the forecast for the next 12 months using Linear Regression.")
                
                # Line plot of historical and future sales
                fig_sales_forecast_lr = px.line(forecast_data_lr, x='date', y=['Actual Sales', 'Predicted Sales'], title='Sales Forecast (Linear Regression)', labels={'value': 'Sales', 'variable': 'Type'}, color_discrete_sequence=px.colors.qualitative.Pastel)
                
                # Display the sales forecast graph and table
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_sales_forecast_lr, use_container_width=True)
                with col2:
                    st.dataframe(forecast_data_lr, height=400)
                
                # Explanation
                st.markdown("""
                **Sales Forecasting (Linear Regression) Insights:**
                - The line plot shows historical sales data (Actual Sales) and the forecasted sales for the next 12 months (Predicted Sales) using the Linear Regression model.
                - The x-axis represents the date, while the y-axis represents the sales amount.
                - The table on the right provides detailed forecast data, including both actual and predicted sales values.
                - This forecast helps in understanding future sales trends and making informed business decisions.
                """)
            except Exception as e:
                st.error(f"Error in sales forecasting: {e}")
# Footer
st.markdown("---")
st.write("Developed by A-Line Business Intelligence Team")