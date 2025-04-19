import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_data(file_path):
    """
    Load the dataset and perform initial processing
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Display basic information
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        
        # Handle missing values (you can customize this based on your strategy)
        df = handle_missing_values(df)
    else:
        print("No missing values found!")
    
    # Save a copy of the processed data
    df.to_csv('data/processed_data/processed_retail_data.csv', index=False)
    
    # Generate basic exploratory plots
    generate_exploratory_plots(df)
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def generate_exploratory_plots(df):
    """
    Generate exploratory data analysis plots
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Create a directory for plots if it doesn't exist
    import os
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Purchase Amount (USD)'], kde=True)
    plt.title('Distribution of Purchase Amount')
    plt.savefig('results/plots/purchase_amount_distribution.png')
    plt.close()
    
    # Plot purchase amount by category
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Category', y='Purchase Amount (USD)', data=df)
    plt.title('Purchase Amount by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/purchase_by_category.png')
    plt.close()
    
    # Plot purchase amount by gender
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Gender', y='Purchase Amount (USD)', data=df)
    plt.title('Purchase Amount by Gender')
    plt.savefig('results/plots/purchase_by_gender.png')
    plt.close()
    
    # Correlation heatmap for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('results/plots/correlation_heatmap.png')
    plt.close()
    
    print(f"Exploratory plots saved to 'results/plots/'")