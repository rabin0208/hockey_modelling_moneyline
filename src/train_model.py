"""
Train a simple logistic regression model for predicting NHL game outcomes.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(data_dir="data"):
    """Load the game features CSV."""
    filepath = os.path.join(data_dir, "game_features.csv")
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} games")
    return df


def prepare_features(df):
    """Prepare features and target variable."""
    # Feature columns (exclude identifiers and target)
    exclude_cols = [
        'game_id', 'date', 'season', 'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name', 'home_wins', 'home_score', 'away_score'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['home_wins'].copy()
    
    print(f"\nFeatures used ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
    
    return X, y, feature_cols


def train_model(X, y, test_size=0.2, random_state=42):
    """Train logistic regression model."""
    print(f"\nSplitting data: {1-test_size:.0%} train, {test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} games")
    print(f"  Test set: {len(X_test)} games")
    
    print("\nTraining logistic regression model with StandardScaler pipeline...")
    # Create pipeline with StandardScaler and LogisticRegression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        )
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.2%}")
    print(f"  Test Accuracy: {test_accuracy:.2%}")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Away Win', 'Home Win']))
    
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  True Negatives (Away Win predicted correctly): {cm[0][0]}")
    print(f"  False Positives (Home Win predicted, but Away won): {cm[0][1]}")
    print(f"  False Negatives (Away Win predicted, but Home won): {cm[1][0]}")
    print(f"  True Positives (Home Win predicted correctly): {cm[1][1]}")
    
    return model, X_test, y_test


def save_model(model, feature_cols, model_dir="models"):
    """Save the trained model as a pickle file."""
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"\nCreated {model_dir}/ folder")
    
    # Save model (pipeline includes scaler and classifier)
    model_file = os.path.join(model_dir, "logistic_regression_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,  # This is a Pipeline with StandardScaler + LogisticRegression
            'feature_columns': feature_cols
        }, f)
    
    print(f"\n✓ Model saved to {model_file}")
    print(f"  Model includes: StandardScaler + LogisticRegression pipeline")
    return model_file


def main():
    """Main function to train and save the model."""
    print("=" * 60)
    print("NHL Game Prediction Model Training")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Save model
    model_file = save_model(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("Model training complete!")
    print(f"Model saved to: {model_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

