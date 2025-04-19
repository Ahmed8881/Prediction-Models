import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

def engineer_features(df):
    """
    Engineer features from the raw data and prepare for modeling
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    # Drop ID column if it exists as it's not a feature
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)
    
    # Separate features and target
    X = df.drop('Purchase Amount (USD)', axis=1)
    y = df['Purchase Amount (USD)']
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Preprocessor saved to 'models/preprocessor.pkl'")
    
    # Get feature names after preprocessing
    feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)
    
    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def get_feature_names(preprocessor, numerical_features, categorical_features):
    """
    Get feature names after preprocessing
    
    Args:
        preprocessor (ColumnTransformer): The fitted preprocessor
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        list: List of feature names after preprocessing
    """
    # Get feature names for numerical features (these remain the same)
    numerical_feature_names = numerical_features
    
    # Get feature names for categorical features (one-hot encoded)
    try:
        categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    except:
        # For older versions of scikit-learn
        categorical_feature_names = []
        for i, category in enumerate(categorical_features):
            n_values = len(preprocessor.named_transformers_['cat'].categories_[i])
            for j in range(n_values):
                categorical_feature_names.append(f"{category}_{j}")
    
    # Combine all feature names
    feature_names = numerical_feature_names + categorical_feature_names
    
    return feature_names