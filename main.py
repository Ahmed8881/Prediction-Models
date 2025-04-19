import os
import pandas as pd
from src.data_processing import load_and_process_data
from src.feature_engineering import engineer_features
from src.model_training import train_all_models
from src.model_evaluation import evaluate_models, plot_model_comparison

def main():
    print("Starting Retail Customer Analytics Pipeline...")
    
    # Create necessary directories if they don't exist
    for directory in ['data/processed_data', 'models', 'results']:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Load and process data
    print("Loading and processing data...")
    df = load_and_process_data('data/retail_customer_data.csv')
    
    # Step 2: Engineer features
    print("Engineering features...")
    X_train, X_test, y_train, y_test, preprocessor = engineer_features(df)
    
    # Step 3: Train models
    print("Training models...")
    models = train_all_models(X_train, y_train)
    
    # Step 4: Evaluate models
    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Step 5: Plot comparison
    print("Generating comparison plots...")
    plot_model_comparison(results)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()