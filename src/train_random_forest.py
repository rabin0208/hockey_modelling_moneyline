"""
Train a Random Forest model for NHL game prediction.

This script loads game features, trains a Random Forest classifier,
evaluates its performance, and saves the trained model.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(filepath='data/game_features.csv'):
    """Load game features from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Features file not found: {filepath}")
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} games")
    
    return df


def prepare_features(df):
    """Prepare features and target for training."""
    # Define feature columns (exclude metadata and target)
    exclude_cols = [
        'game_id', 'date', 'season', 'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name', 'home_wins', 'home_score', 'away_score'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['home_wins'].astype(int)
    
    # Drop rows with any NaN values (only ~2% of data)
    # This ensures consistency with logistic regression models
    before_drop = len(X)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    after_drop = len(X)
    
    if before_drop != after_drop:
        print(f"\nDropped {before_drop - after_drop} games with missing values ({((before_drop - after_drop)/before_drop)*100:.1f}%)")
        print(f"  Using {after_drop} games with complete data")
    
    print(f"\nFeatures used ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
    
    return X, y, feature_cols


def train_random_forest(X_train, y_train, random_state=42, **kwargs):
    """Train a Random Forest classifier with regularization to prevent overfitting."""
    print("\nTraining Random Forest model...")
    
    # Default Random Forest parameters (more conservative to prevent overfitting)
    default_params = {
        'n_estimators': 100,
        'max_depth': 8,  # Further reduced to prevent overfitting
        'min_samples_split': 30,  # Increased to require more samples per split
        'min_samples_leaf': 15,  # Increased to require more samples per leaf
        'max_features': 'sqrt',  # Use sqrt of features (good default)
        'max_samples': 0.7,  # Use 70% of samples per tree (more regularization)
        'random_state': random_state,
        'n_jobs': -1,
        'verbose': 0
    }
    
    # Override with any provided kwargs
    default_params.update(kwargs)
    
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    
    print("✓ Model trained")
    print(f"  Parameters: max_depth={default_params['max_depth']}, "
          f"min_samples_split={default_params['min_samples_split']}, "
          f"min_samples_leaf={default_params['min_samples_leaf']}")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance."""
    print("\nEvaluating model...")
    
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.2%}")
    print(f"  Test Accuracy: {test_accuracy:.2%}")
    
    # Classification report
    print("\nTest Set Classification Report:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=['Away Win', 'Home Win'],
        digits=2
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(f"  True Negatives (Away Win predicted correctly): {cm[0, 0]}")
    print(f"  False Positives (Home Win predicted, but Away won): {cm[0, 1]}")
    print(f"  False Negatives (Away Win predicted, but Home won): {cm[1, 0]}")
    print(f"  True Positives (Home Win predicted correctly): {cm[1, 1]}")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_test_pred': y_test_pred
    }


def save_model(model, filepath='models/random_forest_model.pkl'):
    """Save trained model to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved to {filepath}")


def get_feature_importance(model, feature_names, top_n=10):
    """Get and display top feature importances."""
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:top_n], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    return feature_importance


def main():
    """Main function to train Random Forest model."""
    print("=" * 60)
    print("NHL Game Prediction Model - Random Forest Training")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Split data
    print("\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} games")
    print(f"  Test set: {len(X_test)} games")
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Show feature importance
    feature_importance = get_feature_importance(model, feature_cols)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("Model training complete!")
    print(f"Model saved to: models/random_forest_model.pkl")
    print("=" * 60)


if __name__ == '__main__':
    main()

