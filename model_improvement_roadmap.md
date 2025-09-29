# 🚀 Model Improvement Roadmap

We’ve already built a **solid feature set**:
- ✅ Lagged log returns  
- ✅ High–low volatility spreads  
- ✅ Volume / open interest dynamics  
- ✅ Rolling stats (trend + volatility)  
- ✅ FX log returns  

This is the **backbone most competitors rely on**.  
Now, the **low-hanging fruit** lies in *training & evaluation strategies*.  

---

## 📌 Next Steps & Checkpoints

### 🔹 1. Target Handling
- [ ] **Audit NaNs** across all targets  
- [ ] **Drop extremely sparse targets** (>70% missing)  
- [ ] **Focus training** on denser targets (more reliable signals)  

👉 Benefit: Removes noise that drags down Sharpe ratio  

---

### 🔹 2. Model Objectives
- [ ] Switch from `regression` → `regression_l1` (better rank alignment)  
- [ ] Try `objective="huber"` for robustness to outliers  

👉 Benefit: Optimizes closer to our Sharpe-like evaluation  

---

### 🔹 3. Hyperparameter Tuning
- [ ] Tune key params with **Optuna**:
  - `num_leaves`, `max_depth` → complexity  
  - `min_child_samples` → overfitting control  
  - `subsample`, `colsample_bytree` → randomness/generalization  
  - `reg_alpha`, `reg_lambda` → regularization  

👉 **Next concrete step**: Run Optuna on **`target_0`** and reuse best params across all  

---

### 🔹 4. Cross-Validation Strategy
- [ ] Replace 3-fold `TimeSeriesSplit` with **walk-forward validation**  
  - Train on `[0:T]` → validate on `[T+1:T+k]`  
  - Slide window forward  

👉 Benefit: Mimics real trading, avoids lookahead bias  

---

### 🔹 5. Multi-Target Training
- [ ] Test `MultiOutputRegressor(LGBMRegressor())`  
- [ ] Prototype a **shared-backbone neural net** (shared layers → multiple output heads)  

👉 Benefit: Learns **cross-target correlations**  

---

### 🔹 6. Efficiency
- [ ] Parallelize training with `joblib` / multiprocessing  
- [ ] Experiment with multi-output NN (1 model instead of 400+)  

👉 Benefit: Saves time, scales better  

---

## ⚡ Action Plan
✅ **Today**: Run Optuna tuning loop on `target_0`  
⬜ **Next**: Apply tuned params across all targets  
⬜ **Then**: Upgrade CV strategy to walk-forward  
⬜ **Later**: Explore multi-target NN for efficiency & correlation learning  

---

📊 **Goal**: Improve Sharpe ratio by reducing noise, aligning training with evaluation, and scaling efficiently.  
