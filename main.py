import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Tkagg')
import ta
# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    df = pd.read_csv('Tesla.csv')
    df = df.drop(['Adj Close'], axis=1)

    # Feature engineering
    df['day'] = df['Date'].str.split('/', expand=True)[1].astype(int)
    df['month'] = df['Date'].str.split('/', expand=True)[0].astype(int)
    df['year'] = df['Date'].str.split('/', expand=True)[2].astype(int)
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['ema'] = ta.trend.EMAIndicator(df['Close']).ema_indicator()
    df = df.dropna()

    # Select features and target
    features = df[['open-close', 'low-high', 'is_quarter_end', 'rsi', 'macd', 'ema']]
    target = df['target']

    # Feature scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Time-based split (to avoid data leakage)
    split_idx = int(len(df) * 0.9)
    X_train, X_valid = features[:split_idx], features[split_idx:]
    Y_train, Y_valid = target[:split_idx], target[split_idx:]

    # Handling class imbalance
    ratio = Y_train.value_counts()[0] / Y_train.value_counts()[1]

    # XGBClassifier with regularization and early stopping
    model = XGBClassifier(
        learning_rate=0.05,
        n_estimators=500,
        max_depth=4,
        min_child_weight=5,
        gamma=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        scale_pos_weight=ratio,
        use_label_encoder=False
    )

    model.fit(
        X_train, Y_train,
        eval_set=[(X_valid, Y_valid)],
        verbose=True
    )

    # Evaluation
    y_probs = model.predict_proba(X_valid)[:, 1]
    y_preds = model.predict(X_valid)

    print("Validation ROC-AUC:", metrics.roc_auc_score(Y_valid, y_probs))
    print("Confusion Matrix:\n", metrics.confusion_matrix(Y_valid, y_preds))
    print("Classification Report:\n", metrics.classification_report(Y_valid, y_preds))

    # Confusion Matrix Plot
    metrics.ConfusionMatrixDisplay.from_estimator(model, X_valid, Y_valid)
    plt.title("Confusion Matrix - XGBoost")
    plt.show()
