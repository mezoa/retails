import numpy as np
import pandas as pd
import os

# Step 1: Load the dataset
data_path = 'C:/Users/Meo Angelo Alcantara/Documents/1 CSU/2 Information Technology/4th Year/1st Sem/IS 107/Final Project/Super Store Dataset/train.csv'
data = pd.read_csv(data_path)

# Display basic information about the dataset
print("First 5 rows of the dataset:")
print(data.head())

print("\nShape of the dataset:")
print(data.shape)

print("\nStatistical summary of the dataset:")
print(data.describe())

print("\nColumn names in the dataset:")
print(data.columns)

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum().sort_values(ascending=False))

# Display rows with missing values
print("\nRows with missing values:")
print(data[data.isnull().any(axis=1)])

# Step 2: Clean the dataset
# Drop rows with any missing values
cleaned_data = data.dropna()

print("\nShape of the cleaned dataset:")
print(cleaned_data.shape)

# Display all columns in the DataFrame
pd.set_option('display.max_columns', None)

# Display the first 3 rows of the cleaned DataFrame
print("\nFirst 3 rows of the cleaned dataset:")
print(cleaned_data.head(3))

# Step 3: Convert 'Order Date' and 'Ship Date' columns to datetime
cleaned_data['Order Date'] = pd.to_datetime(cleaned_data['Order Date'], format='%d/%m/%Y', errors='coerce')
cleaned_data['Ship Date'] = pd.to_datetime(cleaned_data['Ship Date'], format='%d/%m/%Y', errors='coerce')

# Drop rows with invalid dates
cleaned_data = cleaned_data.dropna(subset=['Order Date', 'Ship Date'])

# Step 4: Convert 'Postal Code' column to int
cleaned_data['Postal Code'] = cleaned_data['Postal Code'].astype('Int64')  # Use 'Int64' to handle NaNs if any

# Step 5: Insert additional columns for further analysis
# Add columns for order and ship month/year, day, month, and year
cleaned_data['order_month_year'] = cleaned_data['Order Date'].dt.to_period('M')
cleaned_data['ship_month_year'] = cleaned_data['Ship Date'].dt.to_period('M')
cleaned_data['order_day'] = cleaned_data['Order Date'].dt.day
cleaned_data['order_month'] = cleaned_data['Order Date'].dt.month
cleaned_data['order_year'] = cleaned_data['Order Date'].dt.year
cleaned_data['ship_day'] = cleaned_data['Ship Date'].dt.day
cleaned_data['ship_month'] = cleaned_data['Ship Date'].dt.month
cleaned_data['ship_year'] = cleaned_data['Ship Date'].dt.year

# Step 6: Save the cleaned DataFrame to a new CSV file
cleaned_csv_path = 'C:/Users/Meo Angelo Alcantara/Documents/1 CSU/2 Information Technology/4th Year/1st Sem/IS 107/Final Project/Super Store Dataset/cleaned_converted.csv'
cleaned_data.to_csv(cleaned_csv_path, index=False)

# Step 7: Load the cleaned CSV file to verify
cld = pd.read_csv(cleaned_csv_path)

print("\nInformation about the cleaned dataset:")
cld.info()

print("\nShape of the cleaned dataset:")
print(cld.shape)
