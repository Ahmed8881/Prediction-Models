import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('retail_customer_data.csv')

# Examine the data
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Explore the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['Purchase Amount (USD)'])
plt.title('Distribution of Purchase Amount')
plt.show()

# Explore relationships between features and target
plt.figure(figsize=(12, 8))
sns.boxplot(x='Category', y='Purchase Amount (USD)', data=df)
plt.title('Purchase Amount by Category')
plt.xticks(rotation=45)
plt.show()