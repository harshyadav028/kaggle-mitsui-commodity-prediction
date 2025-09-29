# Mitsui Commodity Prediction Challenge

Short: baseline + feature engineering + CV experiments for the Kaggle competition.

Structure:
- notebooks/: exploratory and modeling notebooks
- src/: reusable functions
- data/: raw & processed data (NOT in repo)
- submissions/: CSV submissions

How to run:
1. Follow ENVIRONMENT.md to setup (Colab recommended)
2. Put kaggle.json in ~/.kaggle and run the dataset download from the competition page. See DATA.md

Note: Refer to data and final lag-features data in drive link below.

## EDA
Quick recap (what you just explored)
You checked shapes, date coverage, duplicates.
You visualized missingness across columns and dates.
You grouped instruments by exchange prefix.
You inspected returns behavior and volatility.
You recomputed a target from target_pairs.csv and matched it to labels.
You looked at target autocorr / cross-corr.
You verified test/is_scored and set up leakage-safe lags.

now X is a matrix which has leage safe features and Y is target labels.

Feature Engineering Ideas:
a) Volatility features
From High–Low–Close ranges:

b) Volume & open interest
These often correlate with momentum:

c) Rolling statistics
Capture trends:

d) FX features
Keep them as log returns too (they’re basically time series):

**Project Drive Link: https://drive.google.com/drive/folders/1FpT-HEWExrnK4BhOlgPKpAjk9Gl4CjgO?usp=sharing**`
