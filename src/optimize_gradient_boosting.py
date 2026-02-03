"""
Optimize Gradient Boosting hyperparameters using RandomizedSearchCV.

This script performs hyperparameter optimization for the Histogram-based
Gradient Boosting model to find the best combination of parameters that
reduces overfitting and improves generalization.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    exclude_cols = [
        'game_id', 'date', 'season', 'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name', 'home_wins', 'home_score', 'away_score'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['home_wins'].astype(int)

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


def optimize_hyperparameters(X, y, cv=5, random_state=42, n_iter=50):
    """
    Optimize Gradient Boosting hyperparameters using RandomizedSearchCV.

    Args:
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        n_iter: Number of parameter combinations to try

    Returns:
        Best model, search results, X_test, y_test
    """
    print("\n" + "=" * 60)
    print("Starting Randomized Search with Cross-Validation...")
    print("=" * 60)

    print("\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"  Training set: {len(X_train)} games")
    print(f"  Test set: {len(X_test)} games")

    # Hyperparameter search space for HistGradientBoostingClassifier
    param_distributions = {
        'max_iter': [100, 150, 200, 300],
        'max_depth': [4, 5, 6, 8, 10],
        'min_samples_leaf': [10, 15, 20, 30, 50],
        'l2_regularization': [0.01, 0.1, 0.5, 1.0],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_bins': [128, 255],
        'early_stopping': [True],
        'validation_fraction': [0.1],
        'n_iter_no_change': [10, 15, 20],
    }

    total_combos = 1
    for key, values in param_distributions.items():
        total_combos *= len(values)

    print(f"\n  Cross-validation folds: {cv}")
    print(f"  Parameter combinations to try: {n_iter}")
    print(f"  Total possible combinations: {total_combos}")
    print(f"  (Randomized search samples {n_iter} random combinations)")
    print("\n  This may take a few minutes...")

    base_model = HistGradientBoostingClassifier(
        random_state=random_state,
        verbose=0
    )

    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    print("\n" + "=" * 60)
    print("Randomized Search Results")
    print("=" * 60)
    print(f"\nBest Cross-Validation Score: {random_search.best_score_:.4f} ({random_search.best_score_*100:.2f}%)")
    print("\nBest Hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    print("\n" + "=" * 60)
    print("Evaluating Best Model on Test Set")
    print("=" * 60)

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Overfitting gap: {(train_accuracy - test_accuracy)*100:.2f}%")

    print("\nTest Set Classification Report:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=['Away Win', 'Home Win'],
        digits=2
    ))

    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(f"  True Negatives (Away Win predicted correctly): {cm[0, 0]}")
    print(f"  False Positives (Home Win predicted, but Away won): {cm[0, 1]}")
    print(f"  False Negatives (Away Win predicted, but Home won): {cm[1, 0]}")
    print(f"  True Positives (Home Win predicted correctly): {cm[1, 1]}")

    print("\n" + "=" * 60)
    print("Top 5 Parameter Combinations")
    print("=" * 60)

    results_df = pd.DataFrame(random_search.cv_results_)
    top_results = results_df.nlargest(5, 'mean_test_score')

    for rank, (idx, row) in enumerate(top_results.iterrows(), 1):
        print(f"\n  Rank {rank}:")
        print(f"    Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        print(f"    Train Score: {row['mean_train_score']:.4f}")
        print(f"    Params: {row['params']}")

    return best_model, random_search, X_test, y_test


def save_model(model, filepath='models/gradient_boosting_optimized.pkl'):
    """Save optimized model to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✓ Optimized model saved to {filepath}")


def get_feature_importance(model, feature_names, top_n=10):
    """Get and display top feature importances (if supported by model)."""
    if not hasattr(model, 'feature_importances_'):
        print("\nFeature importance not available for this model.")
        return []
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:top_n], 1):
        print(f"  {i}. {feature}: {importance:.4f}")

    return feature_importance


def main():
    """Main function to optimize Gradient Boosting hyperparameters."""
    print("=" * 60)
    print("NHL Game Prediction Model - Gradient Boosting Hyperparameter Optimization")
    print("=" * 60)

    df = load_data()
    X, y, feature_cols = prepare_features(df)

    best_model, search_results, X_test, y_test = optimize_hyperparameters(X, y)

    get_feature_importance(best_model, feature_cols)

    save_model(best_model)

    print("\n" + "=" * 60)
    print("Hyperparameter optimization complete!")
    print("Optimized model saved to: models/gradient_boosting_optimized.pkl")
    print("=" * 60)


if __name__ == '__main__':
    main()
