# -*- coding: utf-8 -*-
"""
Model development module for insurance enrollment prediction.

This module handles model training, evaluation, and prediction
for the insurance enrollment prediction task.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os

def train_baseline_models(X_train, y_train, cv=5):
    """
    Train multiple baseline models and compare their performance.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Dictionary of trained models and their cross-validation scores
    """
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # Fit the model on the full training set
        model.fit(X_train, y_train)
        
        # Store results
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        
        print(f"{name} CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance using various metrics.
    
    Parameters:
    -----------
    model : estimator
        Trained model to evaluate
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    model_name : str, default="Model"
        Name of the model for display purposes
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Print evaluation results
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['avg_precision']:.4f}")
    
    return metrics

def tune_hyperparameters(X_train, y_train, model_type='random_forest', cv=5):
    """
    Perform hyperparameter tuning for the specified model type.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    model_type : str, default='random_forest'
        Type of model to tune ('random_forest', 'gradient_boosting', or 'logistic_regression')
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    estimator
        Best model after hyperparameter tuning
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    else:  # logistic_regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    
    # Perform grid search
    print(f"\nPerforming hyperparameter tuning for {model_type}...")
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the model.
    
    Parameters:
    -----------
    model : estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature names and their importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def save_model(model, model_path, preprocessor=None, preprocessor_path=None):
    """
    Save trained model and preprocessor to disk.
    
    Parameters:
    -----------
    model : estimator
        Trained model to save
    model_path : str
        Path where to save the model
    preprocessor : object, default=None
        Data preprocessor to save
    preprocessor_path : str, default=None
        Path where to save the preprocessor
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save preprocessor if provided
    if preprocessor is not None and preprocessor_path is not None:
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")

def load_model(model_path, preprocessor_path=None):
    """
    Load trained model and preprocessor from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    preprocessor_path : str, default=None
        Path to the saved preprocessor
        
    Returns:
    --------
    tuple
        (model, preprocessor) if preprocessor_path is provided, otherwise just model
    """
    # Load model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load preprocessor if path is provided
    if preprocessor_path is not None:
        preprocessor = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded from {preprocessor_path}")
        return model, preprocessor
    
    return model

def predict_enrollment(model, preprocessor, data):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    preprocessor : object
        Data preprocessor
    data : pd.DataFrame
        New data for prediction
        
    Returns:
    --------
    np.array
        Predicted probabilities of enrollment
    """
    # Preprocess the data
    X_processed = preprocessor.transform(data)
    
    # Make predictions
    probabilities = model.predict_proba(X_processed)[:, 1]
    
    return probabilities

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_data, engineer_features, preprocess_data
    
    # Load and preprocess data
    df = load_data('../employee_data.csv')
    df_engineered = engineer_features(df)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_engineered)
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Train baseline models
    baseline_results = train_baseline_models(X_train_processed, y_train)
    
    # Select best model from baseline
    best_model_name = max(baseline_results, key=lambda k: baseline_results[k]['mean_cv_score'])
    best_baseline_model = baseline_results[best_model_name]['model']
    print(f"\nBest baseline model: {best_model_name}")
    
    # Evaluate best baseline model
    baseline_metrics = evaluate_model(best_baseline_model, X_test_processed, y_test, model_name=best_model_name)
    
    # Tune hyperparameters for the best model type
    if best_model_name == 'Random Forest':
        model_type = 'random_forest'
    elif best_model_name == 'Gradient Boosting':
        model_type = 'gradient_boosting'
    elif best_model_name == 'Logistic Regression':
        model_type = 'logistic_regression'
    else:
        model_type = 'random_forest'  # Default to Random Forest
    
    # Tune hyperparameters
    tuned_model = tune_hyperparameters(X_train_processed, y_train, model_type=model_type)
    
    # Evaluate tuned model
    tuned_metrics = evaluate_model(tuned_model, X_test_processed, y_test, model_name=f"Tuned {model_type}")
    
    # Get feature importance for the tuned model
    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback for older scikit-learn versions
        from sklearn.preprocessing import OneHotEncoder
        
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat' and isinstance(transformer.named_steps.get('onehot', None), OneHotEncoder):
                feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(columns))
    
    feature_importance = get_feature_importance(tuned_model, feature_names)
    
    # Save the best model
    save_model(
        tuned_model, 
        '../src/models/best_model.joblib', 
        preprocessor, 
        '../src/models/preprocessor.joblib'
    )
