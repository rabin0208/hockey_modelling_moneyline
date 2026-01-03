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

Fetch game data for the last two seasons:

```bash
python src/fetch_historical_seasons.py
```

This will download all games from the past two NHL seasons and save them to:
- `data/games_YYYY_YYYY.json` (individual season files)
- `data/games_all_seasons.json` (combined file)

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
- Train a logistic regression model
- Evaluate performance
- Save the model to `models/logistic_regression_model.pkl`
