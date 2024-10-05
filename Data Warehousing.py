import pandas as pd
import psycopg2
from psycopg2 import sql

# Establish connection
try:
    conn = psycopg2.connect(
        dbname="Retail",
        user="postgres",
        password="admin",
        host="localhost",
        port="5423"
    )
    cur = conn.cursor()
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

# Create tables
try:
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_product (
        product_id VARCHAR PRIMARY KEY,
        product_name VARCHAR,
        category VARCHAR,
        sub_category VARCHAR
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_customer (
        customer_id VARCHAR PRIMARY KEY,
        customer_name VARCHAR,
        segment VARCHAR,
        country VARCHAR,
        city VARCHAR,
        state VARCHAR,
        postal_code VARCHAR,
        region VARCHAR
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_time (
        time_id SERIAL PRIMARY KEY,
        order_date DATE,
        ship_date DATE,
        year INT,
        month INT,
        day INT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS fact_sales (
        sales_id SERIAL PRIMARY KEY,
        order_id VARCHAR,
        product_id VARCHAR,
        customer_id VARCHAR,
        time_id INT,
        ship_mode VARCHAR,
        sales FLOAT,
        FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
        FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
        FOREIGN KEY (time_id) REFERENCES dim_time(time_id)
    );
    """)
except Exception as e:
    print(f"Error creating tables: {e}")
    conn.rollback()
    exit()

# Read cleaned CSV file
try:
    data = pd.read_csv('cleaned_converted.csv')

    # Convert date columns to datetime
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data['Ship Date'] = pd.to_datetime(data['Ship Date'])
except Exception as e:
    print(f"Error reading or processing CSV: {e}")
    conn.rollback()
    exit()

# Insert data into dimension tables in batch
try:
    # Insert into dim_product
    product_data = data[['Product ID', 'Product Name', 'Category', 'Sub-Category']].drop_duplicates()
    product_tuples = [tuple(x) for x in product_data.values]
    cur.executemany("""
        INSERT INTO dim_product (product_id, product_name, category, sub_category)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (product_id) DO NOTHING
    """, product_tuples)

    # Insert into dim_customer
    customer_data = data[['Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region']].drop_duplicates()
    customer_tuples = [tuple(x) for x in customer_data.values]
    cur.executemany("""
        INSERT INTO dim_customer (customer_id, customer_name, segment, country, city, state, postal_code, region)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id) DO NOTHING
    """, customer_tuples)

    # Insert into dim_time
    time_data = data[['Order Date', 'Ship Date']].drop_duplicates()
    time_data['Year'] = time_data['Order Date'].dt.year
    time_data['Month'] = time_data['Order Date'].dt.month
    time_data['Day'] = time_data['Order Date'].dt.day
    time_tuples = [tuple(x) for x in time_data[['Order Date', 'Ship Date', 'Year', 'Month', 'Day']].values]
    cur.executemany("""
        INSERT INTO dim_time (order_date, ship_date, year, month, day)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, time_tuples)

    # Insert fact_sales data
    for index, row in data.iterrows():
        # Get time_id from dim_time
        cur.execute("""
            SELECT time_id FROM dim_time WHERE order_date = %s AND ship_date = %s
        """, (row['Order Date'], row['Ship Date']))
        time_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO fact_sales (order_id, product_id, customer_id, time_id, ship_mode, sales)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (row['Order ID'], row['Product ID'], row['Customer ID'], time_id, row['Ship Mode'], row['Sales']))

    # Commit the transaction
    conn.commit()
except Exception as e:
    print(f"Error inserting data: {e}")
    conn.rollback()

# Close the cursor and connection
cur.close()
conn.close()
