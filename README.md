# Hockey Game Prediction Project

A machine learning project to predict NHL hockey game outcomes.

## Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate hockey_modelling
```

### Using pip

```bash
pip install -r requirements.txt
```

## Test Connection

First, test if you can connect to the NHL API:

```bash
python test_connection.py
```

If the connection works, you'll see a list of NHL teams. If it fails, check your network connection and DNS settings.

## Fetch Data

Fetch NHL data and save it to the `data/` folder:

```bash
python src/fetch_data.py
```

This will download:
- Teams data (`data/teams.json`)
- Current standings (`data/standings.json`)
- Recent schedule (`data/schedule_YYYYMMDD.json`)

## Fetch Historical Seasons

Fetch game data for the last five seasons:

```bash
python src/fetch_historical_seasons.py
```

This will download all games from the past five NHL seasons and save them to:
- `data/games_YYYY_YYYY.json` (individual season files)
- `data/games_all_seasons.json` (combined file)

**Note:** This will take longer to run as it fetches 5 seasons of data (approximately 5,000+ games).

## Create Features

Process the historical games and create features for machine learning:

```bash
python src/create_features.py
```

This will create a CSV file with features for each game:
- `data/game_features.csv`

Features include team win percentages, goal differentials, recent form, and more.

## Update Data with Latest Information

To update your dataset with today's games and latest statistics:

### 1. Update Natural Stat Trick Data (Optional but Recommended)

Natural Stat Trick provides advanced team-level metrics like Corsi and High Danger Chances.

**How to download:**
1. Navigate to [Natural Stat Trick](https://www.naturalstattrick.com/)
2. Go to **"Games"** → **"All Teams"**
3. Select the time period and use the **"Export to CSV"** button at the bottom of the table
4. Save the file as `data/games.csv` (overwrite the existing file)

**Process the NST data:**
```bash
python src/process_nst_data.py
```

This will create `data/nst_processed.csv` with processed advanced statistics.

### 2. Fetch New Games from NHL API

Fetch the latest games from the NHL API:

```bash
python src/fetch_historical_seasons.py
```

This will fetch any new games and update `data/games_all_seasons.json`.

### 3. Create Updated Features

Process all games (including new ones) and create features:

```bash
python src/create_features.py
```

This will update `data/game_features.csv` with features for all games, including the newly fetched ones.

### 4. Optional: Retrain Model

If you want to include the new data in model training:

```bash
python src/train_model.py
# or
python src/optimize_model.py
# or
python src/train_random_forest.py
# or
python src/optimize_random_forest.py
# or
python src/train_gradient_boosting.py
# or
python src/optimize_gradient_boosting.py
```

**Note:** If you only want to make predictions for upcoming games, you can skip this step and use your existing trained model.

## Train Model

Train a logistic regression model on the features:

```bash
python src/train_model.py
```

This will:
- Load the game features
- Train a logistic regression model with StandardScaler
- Evaluate performance
- Save the model to `models/logistic_regression_model.pkl`

## Optimize Model (Hyperparameter Tuning)

Perform hyperparameter optimization to find the best model parameters:

```bash
python src/optimize_model.py
```

This will:
- Perform randomized search with cross-validation
- Test 50 random combinations of C, penalty, solver, and class_weight
- Find the best hyperparameters
- Save the optimized model to `models/logistic_regression_optimized.pkl`

**Note:** This uses randomized search (faster than grid search) and tests 50 parameter combinations. Takes a few minutes to run.

## Train Random Forest Model

Train a Random Forest classifier (often performs better than logistic regression):

```bash
python src/train_random_forest.py
```

This will:
- Train a Random Forest model with default hyperparameters
- Evaluate performance on test set
- Display feature importance rankings
- Save the model to `models/random_forest_model.pkl`

**Note:** Random Forest can capture non-linear relationships and often achieves better accuracy than logistic regression.

## Optimize Random Forest Model

Perform hyperparameter optimization for the Random Forest model:

```bash
python src/optimize_random_forest.py
```

This will:
- Perform randomized search with cross-validation
- Test 50 random combinations of hyperparameters (n_estimators, max_depth, min_samples_split, etc.)
- Find the best hyperparameters that reduce overfitting
- Save the optimized model to `models/random_forest_optimized.pkl`

**Note:** This uses randomized search and tests 50 parameter combinations. Takes several minutes to run.

## Train Gradient Boosting Model

Train a Histogram-based Gradient Boosting classifier (often performs better than Random Forest):

```bash
python src/train_gradient_boosting.py
```

This will:
- Train a Gradient Boosting model with default hyperparameters
- Evaluate performance on test set
- Save the model to `models/gradient_boosting_model.pkl`

**Note:** Gradient Boosting uses boosting (sequential tree building) instead of bagging, and often achieves better accuracy than Random Forest on tabular data.

## Optimize Gradient Boosting Model

Perform hyperparameter optimization for the Gradient Boosting model:

```bash
python src/optimize_gradient_boosting.py
```

This will:
- Perform randomized search with cross-validation
- Test 50 random combinations of hyperparameters (max_iter, max_depth, learning_rate, l2_regularization, etc.)
- Find the best hyperparameters that reduce overfitting
- Save the optimized model to `models/gradient_boosting_optimized.pkl`

**Note:** This uses randomized search and tests 50 parameter combinations. Takes several minutes to run.

## Predict Upcoming Games

Predict outcomes for upcoming games using a trained model:

```bash
python src/predict_games.py
```

Or specify a different model and time period:

```bash
python src/predict_games.py --model models/logistic_regression_optimized.pkl --days 14
```

This will:
- Load a saved model (default: Random Forest optimized)
- Fetch upcoming games from NHL API
- Create features for each game using historical data
- Make predictions with win probabilities
- Display results with:
  - Predicted winner
  - Win probabilities for both teams
  - Implied odds (American format)
  - Confidence level

**Available models:**
- `models/gradient_boosting_optimized.pkl`
- `models/random_forest_optimized.pkl` (default)
- `models/gradient_boosting_model.pkl`
- `models/logistic_regression_optimized.pkl`
- `models/random_forest_model.pkl`
- `models/logistic_regression_model.pkl`
