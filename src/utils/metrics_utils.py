"""
Metrics Utilities
"""

import numpy as np
import torch


def calculate_regression_metrics(y_true, y_pred):
   
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
   
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
   
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Calculate metrics
    
    #Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    #Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    #Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    #Mean Absolute Percentage Error (MAPE)
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf  
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    mask = denominator != 0
    if np.sum(mask) > 0:
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    else:
        smape = 0.0  
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'smape': float(smape)
    }


def print_metrics(metrics, prefix=""):
    
    title = f"{prefix} Metrics:" if prefix else "Metrics:"
    print(title)
    print(f"  MSE:   {metrics['mse']:.4f}")
    print(f"  RMSE:  {metrics['rmse']:.4f}")
    print(f"  MAE:   {metrics['mae']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  SMAPE: {metrics['smape']:.2f}%")


