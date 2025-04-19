import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models on test data
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary of evaluation results
    """
    # Convert to dense array if sparse (for models that require it)
    if hasattr(X_test, "toarray"):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
        
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Use dense array for Naive Bayes
        if name == "Naive_Bayes":
            X_test_eval = X_test_dense
        else:
            X_test_eval = X_test
        
        # Make predictions
        try:
            y_pred = model.predict(X_test_eval)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"  MSE: {mse:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.2f}")
            
            # Save individual model results
            save_model_results(name, y_test, y_pred)
            
        except Exception as e:
            print(f"  Error evaluating {name}: {e}")
            results[name] = {
                'MSE': float('nan'),
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'R2': float('nan')
            }
    
    # Save all results to CSV
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[model]['MSE'] for model in results],
        'RMSE': [results[model]['RMSE'] for model in results],
        'MAE': [results[model]['MAE'] for model in results],
        'R²': [results[model]['R2'] for model in results]
    })
    
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("Results saved to 'results/model_comparison.csv'")
    
    return results

def save_model_results(model_name, y_true, y_pred):
    """
    Save individual model results and generate plots
    
    Args:
        model_name (str): Name of the model
        y_true: True target values
        y_pred: Predicted target values
    """
    # Create directory if it doesn't exist
    os.makedirs('results/model_evaluations', exist_ok=True)
    
    # Save predictions vs actuals
    pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': y_true - y_pred
    }).to_csv(f'results/model_evaluations/{model_name}_predictions.csv', index=False)
    
    # Create scatter plot of actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'results/model_evaluations/{model_name}_scatter.png')
    plt.close()
    
    # Create error distribution plot
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_name}: Error Distribution')
    plt.tight_layout()
    plt.savefig(f'results/model_evaluations/{model_name}_error_dist.png')
    plt.close()

def plot_model_comparison(results):
    """
    Generate comparison plots for all models
    
    Args:
        results (dict): Dictionary of evaluation results
    """
    # Create directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    
    # Extract model names and performance metrics
    models = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models]
    r2_values = [results[model]['R2'] for model in models]
    
    # Sort models by R² (best to worst)
    sorted_indices = np.argsort(r2_values)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_r2 = [r2_values[i] for i in sorted_indices]
    sorted_mse = [mse_values[i] for i in sorted_indices]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # R² plot (higher is better)
    ax1.barh(sorted_models, sorted_r2, color='green')
    ax1.set_title('Model Comparison: R² Score (higher is better)')
    ax1.set_xlabel('R² Score')
    ax1.set_xlim(0, 1.0)  # R² typically ranges from 0 to 1
    
    # MSE plot (lower is better)
    ax2.barh(sorted_models, sorted_mse, color='red')
    ax2.set_title('Model Comparison: MSE (lower is better)')
    ax2.set_xlabel('Mean Squared Error')
    
    plt.tight_layout()
    plt.savefig('results/plots/model_comparison.png')
    plt.close()
    
    # Create radar chart for comprehensive comparison
    metrics = ['R²', 'MSE', 'RMSE', 'MAE']
    
    # Prepare data for radar chart
    # We need to normalize MSE, RMSE, and MAE (lower is better)
    # and invert them so higher values are better (consistent with R²)
    r2_norm = [results[model]['R2'] for model in models]
    
    # Find max values for normalization
    max_mse = max(results[model]['MSE'] for model in models)
    max_rmse = max(results[model]['RMSE'] for model in models)
    max_mae = max(results[model]['MAE'] for model in models)
    
    # Normalize and invert (1 - value/max) so higher is better
    mse_norm = [1 - results[model]['MSE']/max_mse for model in models]
    rmse_norm = [1 - results[model]['RMSE']/max_rmse for model in models]
    mae_norm = [1 - results[model]['MAE']/max_mae for model in models]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot for each model
    for i, model in enumerate(models):
        values = [r2_norm[i], mse_norm[i], rmse_norm[i], mae_norm[i]]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Comparison (Higher is Better)')
    plt.tight_layout()
    plt.savefig('results/plots/model_radar_comparison.png')
    plt.close()