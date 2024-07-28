import pandas as pd


file_path = 'mobile_sales.csv'
mobile_sales = pd.read_csv(file_path)

print("Missing values before cleaning:\n", mobile_sales.isnull().sum())

mobile_sales = mobile_sales.dropna()  


print("Duplicate rows before cleaning:", mobile_sales.duplicated().sum())

mobile_sales = mobile_sales.drop_duplicates()

mobile_sales['Date'] = pd.to_datetime(mobile_sales['Date'], errors='coerce')

mobile_sales['CustomerGender'] = mobile_sales['CustomerGender'].str.capitalize()


mobile_sales['Price'] = mobile_sales['Price'].round(2)
mobile_sales['TotalRevenue'] = mobile_sales['TotalRevenue'].round(2)

# Display the cleaned data
print("Missing values after cleaning:\n", mobile_sales.isnull().sum())
print("Duplicate rows after cleaning:", mobile_sales.duplicated().sum())
print("Data after cleaning:\n", mobile_sales.head())

# Save the cleaned dataset
mobile_sales.to_csv('cleaned_mobile_sales.csv', index=False)
