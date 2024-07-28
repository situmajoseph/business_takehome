import pandas as pd
import random


file_path = 'mobile_sales.csv'
mobile_sales = pd.read_csv(file_path)


customer_satisfaction_data = {
    'TransactionID': mobile_sales['TransactionID'],
    'CustomerSatisfaction': [random.randint(1, 5) for _ in range(len(mobile_sales))]
}
customer_satisfaction = pd.DataFrame(customer_satisfaction_data)


integrated_data = pd.merge(mobile_sales, customer_satisfaction, on='TransactionID')

integrated_data.to_csv('integrated_mobile_sales.csv', index=False)


print(integrated_data.head())
