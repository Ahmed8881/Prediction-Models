import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

def train_all_models(X_train, y_train):
    """
    Train all 7 requested models
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    # Convert to dense array if sparse
    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train
    
    # Initialize models
    models = {
        "Linear_Regression": train_linear_regression(X_train, y_train),
        "SVM": train_svm(X_train, y_train),
        "Decision_Tree": train_decision_tree(X_train, y_train),
        "Random_Forest": train_random_forest(X_train, y_train),
        "KNN": train_knn(X_train, y_train),
        "Naive_Bayes": train_naive_bayes(X_train_dense, y_train),  # NB requires dense input
        "Neural_Network": train_neural_network(X_train, y_train)
    }
    
    # Save each model
    for name, model in models.items():
        joblib.dump(model, f'models/{name}.pkl')
        print(f"Model '{name}' saved to 'models/{name}.pkl'")
    
    return models

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model"""
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train SVM model"""
    print("Training SVM model...")
    # Using a basic SVR with default parameters for now
    # For large datasets, this might be slow, so we use a subset for initial training
    if X_train.shape[0] > 10000:
        sample_idx = np.random.choice(X_train.shape[0], 10000, replace=False)
        X_sample = X_train[sample_idx]
        y_sample = y_train.iloc[sample_idx] if hasattr(y_train, 'iloc') else y_train[sample_idx]
        model = SVR()
        model.fit(X_sample, y_sample)
    else:
        model = SVR()
        model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model"""
    print("Training Decision Tree model...")
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model with basic hyperparameter tuning"""
    print("Training Random Forest model...")
    # For large datasets, we skip hyperparameter tuning
    if X_train.shape[0] > 10000:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    else:
        # Basic hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        base_model = RandomForestRegressor(random_state=42)
        model = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error')
        model.fit(X_train, y_train)
        print(f"Best parameters for Random Forest: {model.best_params_}")
    
    return model

def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors model"""
    print("Training KNN model...")
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model"""
    print("Training Naive Bayes model...")
    # Note: Naive Bayes is typically used for classification, 
    # but we're using GaussianNB as it can work with continuous data
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    """Train Neural Network model"""
    print("Training Neural Network model...")
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        activation='relu',
        solver='adam',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train, y_train)
    return model