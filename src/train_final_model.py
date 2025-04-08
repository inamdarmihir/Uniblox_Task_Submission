"""
Script to train and save the final model for insurance enrollment prediction.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from data_preprocessing import load_data, engineer_features, preprocess_data
from model import evaluate_model, save_model

def train_final_model():
    """
    Train the final model with best hyperparameters and save it.
    """
    print("Training Final Model")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_data('../employee_data.csv')
    df_engineered = engineer_features(df)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_engineered)
    
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Initialize final model with best hyperparameters
    print("\nTraining final model...")
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # Train the model
    final_model.fit(X_train_processed, y_train)
    
    # Evaluate the model
    print("\nEvaluating final model...")
    evaluate_model(final_model, X_test_processed, y_test, "Final Model")
    
    # Save the model and preprocessor
    print("\nSaving model and preprocessor...")
    save_model(
        final_model,
        'models/final_model.joblib',
        preprocessor=preprocessor,
        preprocessor_path='models/preprocessor.joblib'
    )
    
    print("\nFinal model training complete!")
    print("Model saved to: models/final_model.joblib")
    print("Preprocessor saved to: models/preprocessor.joblib")

if __name__ == "__main__":
    train_final_model() 