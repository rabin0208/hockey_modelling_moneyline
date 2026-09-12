# Model benchmark — 2026-09-12

Held-out accuracy and simulated moneyline ROI by NHL season.

## Setup

| Item | Value |
|---|---|
| Model | Optimized logistic regression (`C=0.001`, L2, StandardScaler pipeline) |
| Features | `data/game_features.csv` (25 columns; rows with missing values dropped) |
| Games | 6,560 complete-feature games, 2021-10-12 to 2026-02-04 |
| Validation | Leave-one-season-out: train on all other seasons, test on the held-out year |
| Bet | $1 moneyline on the model's predicted winner |
| Odds | Kaggle home moneylines; away line inferred as the no-vig complement of the home price |
| Usable lines | American odds with absolute value 101–800; placeholder −105 lines dropped when they dominate a season |

## Season results

| Season | Games | With odds | Accuracy | Home baseline | Market favorite | ROC-AUC | Bets | Profit ($1/bet) | ROI | Value bets | Value ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021-22 | 1,315 | 1,288 | 61.67% | 54.14% | 52.17% | 0.6518 | 1,288 | +344.66 | 26.76% | 973 | 36.50% |
| 2022-23 | 1,451 | 1,422 | 60.17% | 52.72% | 52.74% | 0.6444 | 1,422 | +348.51 | 24.51% | 1,206 | 29.57% |
| 2023-24 | 1,447 | 1,394 | 59.85% | 53.77% | 51.36% | 0.6216 | 1,394 | +322.28 | 23.12% | 1,159 | 28.53% |
| 2024-25 | 1,425 | 0 | 58.88% | 56.35% | — | 0.6091 | 0 | — | — | 0 | — |
| 2025-26 | 922 | 500 | 54.56% | 52.28% | 51.60% | 0.5546 | 500 | +12.33 | 2.47% | 417 | 5.82% |
| **Overall** | **6,560** | **4,604** | **59.33%** | **53.96%** | **52.04%** | **0.6214** | **4,604** | **+1,027.78** | **22.32%** | **3,755** | **28.41%** |

**Home baseline** is always picking the home team. **Market favorite** is picking the side with the negative moneyline. **Value bets** are the subset of model picks where the model's win probability exceeds the implied market probability.

## Model comparison (random 80/20 split)

Same feature matrix, `random_state=42`, 1,312 test games. Used only to choose which algorithm to backtest.

| Model | Test accuracy | ROC-AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| **Logistic regression optimized** | **60.14%** | **0.6525** | **0.6597** | **0.2336** |
| Logistic regression default | 59.68% | 0.6439 | 0.6595 | 0.2336 |
| Random forest default | 59.68% | 0.6360 | 0.6605 | 0.2342 |
| Random forest optimized | 58.77% | 0.6288 | 0.6632 | 0.2355 |
| Gradient boosting default | 58.61% | 0.6170 | 0.6684 | 0.2380 |
| Gradient boosting optimized | 58.00% | 0.6102 | 0.6708 | 0.2390 |

This split mixes years, so it is optimistic relative to the leave-one-season-out table above.

## Notes

- 2024-25 has no usable moneylines: every Kaggle `home_moneyline` is a placeholder −105. Accuracy is reported; ROI is not.
- 2025-26 is a partial season (through 2026-02-04). Only 500 of 922 games have real lines.
- Away moneylines are not in the source file. They are inferred from the home price with no extra juice, which is slightly optimistic for away bets.
- These Kaggle prices are only weakly related to game results, so ROI is a historical simulation, not live sportsbook edge.
- Reproduce with `python src/evaluate_historical.py` (writes `data/historical_accuracy_roi.csv`).
