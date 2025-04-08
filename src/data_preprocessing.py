# -*- coding: utf-8 -*-
"""
Data preprocessing module for insurance enrollment prediction.

This module handles data loading, cleaning, feature engineering, 
and preprocessing for the insurance enrollment prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the employee dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def check_data_quality(df):
    """
    Check data quality issues like missing values and duplicates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with data quality information
    """
    # Check missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Check duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    
    # Create data quality report
    quality_report = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percentage
    })
    
    return quality_report

def preprocess_data(df, target_col='enrolled', test_size=0.2, random_state=42):
    """
    Preprocess the data for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str, default='enrolled'
        Name of the target column
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Remove employee_id as it's not a predictive feature
    if 'employee_id' in df.columns:
        df = df.drop('employee_id', axis=1)
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

def engineer_features(df):
    """
    Create new features based on existing ones.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Age groups
    df_new['age_group'] = pd.cut(
        df_new['age'], 
        bins=[0, 30, 40, 50, 60, 100], 
        labels=['<30', '30-40', '40-50', '50-60', '60+']
    )
    
    # Salary ranges (quintiles)
    df_new['salary_range'] = pd.qcut(
        df_new['salary'], 
        q=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Tenure groups
    df_new['tenure_group'] = pd.cut(
        df_new['tenure_years'], 
        bins=[-0.1, 1, 3, 5, 10, 100], 
        labels=['<1', '1-3', '3-5', '5-10', '10+']
    )
    
    # Interaction features
    # Combine employment type and has_dependents
    df_new['emp_with_dependents'] = df_new['employment_type'] + '_' + df_new['has_dependents']
    
    # Combine region and marital_status
    df_new['region_marital'] = df_new['region'] + '_' + df_new['marital_status']
    
    # Salary per year of tenure (to capture compensation growth)
    df_new['salary_per_tenure'] = df_new['salary'] / (df_new['tenure_years'] + 1)  # Adding 1 to avoid division by zero
    
    print(f"Added {len(df_new.columns) - len(df.columns)} new features")
    return df_new

if __name__ == "__main__":
    # Example usage
    df = load_data('../employee_data.csv')
    quality_report = check_data_quality(df)
    print("\nData Quality Report:")
    print(quality_report)
    
    # Engineer features
    df_engineered = engineer_features(df)
    print("\nEngineered Features:")
    print(df_engineered.columns.tolist())
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_engineered)
