# Tesla Stock Movement Prediction using XGBoost
This project builds a binary classification model to predict the next-day movement (up/down) of Tesla's stock price using technical indicators and engineered features. The core model is an XGBoost classifier trained with historical data.

# Features
Data preprocessing and feature engineering:
Daily price change (open-close, low-high)
Quarter-end flag
Date-based features (day, month, year)
Technical indicators: RSI, MACD, EMA
Handling class imbalance using scale_pos_weight
Train/validation split based on time (no leakage)
Performance metrics: ROC AUC, confusion matrix, classification report
Visual analysis with matplotlib

# Technologies Used
Python
Pandas, NumPy, scikit-learn
XGBoost
TA-Lib (via ta library)
Matplotlib

