# gridirony

Fantasy football prediction machine.

Gridirony uses historical NFL statistics and betting data to predict game outcomes. It includes two models:
- **Linear model**: Predicts the point margin (home team score minus away team score)
- **Logistic model**: Predicts the probability that the home team wins

## Environment Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Initialization

Build the dataset by running:

```powershell
python src/py/build_dataset.py
```

This script:
1. Downloads NFL data (seasons 2009-2024) via the `nfl_data_py` library
2. Saves raw data to `data/raw/` (schedules, weekly stats, betting lines)
3. Engineers rolling window features (3-game and 4-game averages for points, yards, turnovers)
4. Outputs the final feature matrix to `data/features/model_df.parquet`

## Running the Models

**Linear Model** (point margin prediction):
```powershell
python src/py/train_linear.py
```

**Logistic Model** (win probability prediction):
```powershell
python src/py/train_logit.py
```

Both models use elastic-net regularization and are trained on seasons 2009-2019, validated on 2020, and tested on 2021+.

## Understanding the Output

### Output Files

| File | Description |
|------|-------------|
| `reports/linear_test_predictions.csv` | Point spread predictions vs actuals |
| `reports/logit_test_predictions.csv` | Win probability predictions vs actuals |
| `artifacts/*.pkl` | Saved model objects for reuse |

### Interpreting Predictions

**Linear model (`pred_margin`)**:
- Positive value = model predicts home team wins by that many points
- Negative value = model predicts away team wins by that many points
- Example: `pred_margin = 3.5` means home team favored by 3.5 points

**Logistic model (`pred_prob`)**:
- Value between 0 and 1 representing probability the home team wins
- Example: `pred_prob = 0.65` means 65% chance home team wins

### Metrics

**Linear model**:
- **MAE** (Mean Absolute Error): Average prediction error in points
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily

**Logistic model**:
- **Brier Score**: Mean squared error for probabilities (lower is better, 0.25 = random)
- **Log Loss**: Cross-entropy loss (lower is better)
- **ROC-AUC**: Model's ability to rank outcomes (0.5 = random, 1.0 = perfect)

## Project Structure

```
gridirony/
├── src/py/              # Python source code
│   ├── build_dataset.py # Data acquisition and feature engineering
│   ├── train_linear.py  # Linear regression model
│   └── train_logit.py   # Logistic regression model
├── data/
│   ├── raw/             # Raw NFL data (parquet)
│   └── features/        # Engineered features (parquet)
├── artifacts/           # Trained model objects
├── reports/             # Prediction outputs
└── requirements.txt     # Python dependencies
```
