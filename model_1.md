LightGBM is a string starting point for any competeiton

# Model 1: LightGBM with Multi-Target Time-Series Forecasting

## Overview
This notebook builds **Model 1**, where we apply **LightGBM (LGBM)** for regression tasks across multiple targets in a **time-series setting**.  
The dataset consists of hundreds of target variables (≈400), each representing a separate prediction task. The challenge lies in efficiently training and evaluating models across all these targets while respecting the **temporal structure** of the data.

---

## Key Steps

### 1. Data Preparation
- Features (`X`) and multiple targets (`Y`) were extracted from the dataset.  
- Each target column (`target_0`, `target_1`, ..., `target_n`) is treated as a **separate regression task**.  
- The dataset is split into **time-based folds** instead of random folds to avoid leakage from the future into the past.

### 2. Cross-Validation Strategy
- **TimeSeriesSplit** (or similar chronological validation) is used.  
- For each fold:
  - Train on past data
  - Validate on the immediate future segment  
- This simulates a real-world forecasting scenario.

### 3. Model Choice: LightGBM
- We use **LightGBM’s LGBMRegressor** for efficiency on tabular data.  
- Reasons for choosing LightGBM:
  - Handles large feature sets (600+ features in this dataset).  
  - Fast training and prediction.  
  - Supports early stopping to prevent overfitting.

### 4. Handling Multiple Targets
- Instead of a single multi-output model, we train **one LightGBM model per target**.  
- Looping through targets:
  - Extract `(X_train, y_train)` for each target.  
  - Train a LightGBM regressor.  
  - Store predictions and evaluation metrics.  
- This ensures that each target’s model is tailored to its distribution and dynamics.

### 5. Evaluation Metrics
- **RMSE (Root Mean Squared Error)** per fold and per target.  
- **Sharpe Ratio**: measures risk-adjusted return across predictions (important in financial time-series).  
- Observations:
  - Some folds converge in very few boosting rounds (1–5), meaning weak predictive signals.  
  - RMSE is relatively low (≈0.01–0.02), but Sharpe ratio is near 0, implying limited real-world utility.

---

## Observations & Challenges
- Many warnings like *“No further splits with positive gain”* → suggests weak feature-target relationships.  
- Models often stopped training early (sometimes after just 1–2 boosting rounds).  
- This indicates the dataset might not have strong predictive signals for some targets.  
- While RMSE values appear small, **Sharpe ratio shows performance is weak in a financial context**.

---

## Next Steps
1. **Feature Engineering**: Add lag features, rolling means, differences, or domain-driven signals.  
2. **Target Engineering**: Instead of predicting raw returns, predict direction (classification) or smoothed targets.  
3. **Modeling Approaches**:
   - Try **multi-target learning** (e.g., multi-output regressors, neural nets).  
   - Use **regularization or feature selection** to reduce noise.  
   - Explore **temporal models** (e.g., RNNs, Transformers).  
4. **Evaluation**: Focus more on Sharpe ratio and profitability metrics instead of just RMSE.

---

## Conclusion
Model 1 establishes a **baseline**:
- Trains **LightGBM regressors per target** with **time-series CV**.  
- Provides per-target and overall evaluation (RMSE + Sharpe).  
- Shows that while error is numerically small, **financial signal quality is weak**.

This will guide improvements in **Model 2**, where we’ll focus on stronger feature engineering, regularization, and domain-specific evaluation.


#### we tackle multiple target by training one model forn one target 
