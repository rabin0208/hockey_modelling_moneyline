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
