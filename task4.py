import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np

# Load the integrated dataset
file_path = 'integrated_mobile_sales.csv'
data = pd.read_csv(file_path)

# Step 1: Exploratory Data Analysis (EDA)

# Summary statistics
summary_stats = data.describe()
print("Summary Statistics:\n", summary_stats)

# Distribution plots
plt.figure(figsize=(12, 8))
sns.histplot(data['Price'], kde=True, bins=30)
plt.title('Price Distribution')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(data['UnitsSold'], kde=True, bins=30)
plt.title('Units Sold Distribution')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(data['TotalRevenue'], kde=True, bins=30)
plt.title('Total Revenue Distribution')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(data['CustomerSatisfaction'])
plt.title('Customer Satisfaction Distribution')
plt.show()

# Correlation analysis - only include numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = data[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Insights from customer satisfaction
numeric_columns.remove('CustomerSatisfaction')
satisfaction_group = data.groupby('CustomerSatisfaction')[numeric_columns].mean()
print("Average Metrics by Customer Satisfaction:\n", satisfaction_group)

# Step 2: Statistical Analysis - Regression Analysis
X = data[['Price', 'UnitsSold']]
y = data['TotalRevenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) for Regression:", mse)
print("Regression Coefficients:", regressor.coef_)
print("Regression Intercept:", regressor.intercept_)

# Step 3: Machine Learning Techniques - Clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(data[['Price', 'UnitsSold', 'CustomerAge']])

# Visualization of clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Price', y='UnitsSold', hue='Cluster', data=data, palette='viridis')
plt.title('K-Means Clustering')
plt.show()

# Classification - Predicting Customer Satisfaction
X = data[['Price', 'UnitsSold', 'TotalRevenue', 'CustomerAge']]
y = data['CustomerSatisfaction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Model evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
