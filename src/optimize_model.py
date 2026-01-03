"""
Hyperparameter optimization for logistic regression model.

This script performs randomized search to find the best hyperparameters
for the logistic regression model.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    
    # Drop rows with any NaN values (only ~2% of data)
    # This is better than imputing since we have good coverage
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


def optimize_hyperparameters(X, y, cv=5, n_iter=50, random_state=42):
    """
    Perform randomized search to find best hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target variable
        cv: Number of cross-validation folds
        n_iter: Number of parameter combinations to try (default 50)
        random_state: Random seed
    
    Returns:
        Best model and results
    """
    print(f"\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} games")
    print(f"  Test set: {len(X_test)} games")
    
    # Create pipeline
    # Use higher max_iter to avoid convergence warnings, especially for saga solver
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=random_state, max_iter=5000)
    )
    
    # Define hyperparameter search space
    # RandomizedSearchCV will sample random combinations from this space
    # Note: Not all solver/penalty combinations work together:
    # - 'lbfgs' only works with 'l2'
    # - 'liblinear' works with 'l1' and 'l2'
    # - 'saga' works with 'l1', 'l2', and 'elasticnet'
    param_distributions = [
        {
            'logisticregression__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'logisticregression__penalty': ['l2'],
            'logisticregression__solver': ['lbfgs', 'liblinear', 'saga'],
            'logisticregression__class_weight': [None, 'balanced']
        },
        {
            'logisticregression__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'logisticregression__penalty': ['l1'],
            'logisticregression__solver': ['liblinear', 'saga'],
            'logisticregression__class_weight': [None, 'balanced']
        },
        {
            'logisticregression__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'logisticregression__penalty': ['elasticnet'],
            'logisticregression__solver': ['saga'],
            'logisticregression__l1_ratio': [0.1, 0.5, 0.9],  # Required for elasticnet
            'logisticregression__class_weight': [None, 'balanced']
        }
    ]
    
    # Calculate total possible combinations (for reference)
    total_combos = 0
    for param_set in param_distributions:
        combos = 1
        for key, values in param_set.items():
            combos *= len(values)
        total_combos += combos
    
    print("\n" + "=" * 60)
    print("Starting Randomized Search with Cross-Validation...")
    print("=" * 60)
    print(f"  Cross-validation folds: {cv}")
    print(f"  Parameter combinations to try: {n_iter}")
    print(f"  Total possible combinations: {total_combos}")
    print(f"  (Randomized search samples {n_iter} random combinations)")
    print("\n  This may take a few minutes...")
    
    # Create randomized search with scoring
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,  # Parameter space to sample from
        n_iter=n_iter,  # Number of random parameter combinations to try
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all available CPU cores
        verbose=1,
        random_state=random_state,
        return_train_score=True
    )
    
    # Perform randomized search
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print("\n" + "=" * 60)
    print("Randomized Search Results")
    print("=" * 60)
    print(f"\nBest Cross-Validation Score: {best_score:.4f} ({best_score*100:.2f}%)")
    print(f"\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating Best Model on Test Set")
    print("=" * 60)
    
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTest Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Away Win', 'Home Win']))
    
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  True Negatives (Away Win predicted correctly): {cm[0][0]}")
    print(f"  False Positives (Home Win predicted, but Away won): {cm[0][1]}")
    print(f"  False Negatives (Away Win predicted, but Home won): {cm[1][0]}")
    print(f"  True Positives (Home Win predicted correctly): {cm[1][1]}")
    
    # Show top 5 parameter combinations
    print("\n" + "=" * 60)
    print("Top 5 Parameter Combinations")
    print("=" * 60)
    results_df = pd.DataFrame(random_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    for idx, row in top_5.iterrows():
        print(f"\n  Rank {len(top_5) - list(top_5.index).index(idx)}:")
        print(f"    Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f})")
        print(f"    Params: {row['params']}")
    
    return best_model, best_params, random_search, X_test, y_test


def save_model(model, feature_cols, best_params, model_dir="models"):
    """Save the optimized model as a pickle file."""
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"\nCreated {model_dir}/ folder")
    
    # Save model
    model_file = os.path.join(model_dir, "logistic_regression_optimized.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_columns': feature_cols,
            'best_parameters': best_params
        }, f)
    
    print(f"\n✓ Optimized model saved to {model_file}")
    return model_file


def main():
    """Main function to optimize and save the model."""
    print("=" * 60)
    print("NHL Game Prediction Model - Hyperparameter Optimization")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Optimize hyperparameters
    best_model, best_params, random_search, X_test, y_test = optimize_hyperparameters(X, y)
    
    # Save model
    model_file = save_model(best_model, feature_cols, best_params)
    
    print("\n" + "=" * 60)
    print("Hyperparameter optimization complete!")
    print(f"Optimized model saved to: {model_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

