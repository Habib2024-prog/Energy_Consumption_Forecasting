"""
run_once_train_seq_model.py
───────────────────────────
Run this ONCE locally (not on Streamlit Cloud) to generate:
    models/seq_model.pkl

This trains a GradientBoostingRegressor on lag features — a lightweight
replacement for the LSTM that works identically in the Streamlit app
without needing TensorFlow.

Usage:
    python run_once_train_seq_model.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Load your existing assets ─────────────────────────────────
with open('models/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('models/scaler_Y.pkl', 'rb') as f:
    scaler_Y = pickle.load(f)
with open('models/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('models/lstm_config.pkl', 'rb') as f:
    lstm_config = pickle.load(f)

WINDOW_SIZE = lstm_config['window_size']

train_df = pd.read_csv('data/processed/train.csv')
test_df  = pd.read_csv('data/processed/test.csv')

X_train_sc = scaler_X.transform(train_df[feature_cols])
y_train_sc = scaler_Y.transform(train_df[['Appliances']]).ravel()
X_test_sc  = scaler_X.transform(test_df[feature_cols])
y_test_sc  = scaler_Y.transform(test_df[['Appliances']]).ravel()


def make_lag_features(X_scaled, window_size):
    """Flatten a rolling window into a single feature vector per row."""
    rows = []
    for i in range(window_size, len(X_scaled)):
        window = X_scaled[i - window_size:i].ravel()
        rows.append(window)
    return np.array(rows)


print(f"Building lag features (window={WINDOW_SIZE}) …")
X_train_lag = make_lag_features(X_train_sc, WINDOW_SIZE)
y_train_lag = y_train_sc[WINDOW_SIZE:]
X_test_lag  = make_lag_features(X_test_sc,  WINDOW_SIZE)
y_test_lag  = y_test_sc[WINDOW_SIZE:]

print(f"Train shape: {X_train_lag.shape}, Test shape: {X_test_lag.shape}")

# ── Train ─────────────────────────────────────────────────────
print("Training GradientBoostingRegressor …")
seq_model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    random_state=42,
    verbose=1
)
seq_model.fit(X_train_lag, y_train_lag)

# ── Evaluate ──────────────────────────────────────────────────
y_pred_sc = seq_model.predict(X_test_lag)
y_pred = scaler_Y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
y_true = scaler_Y.inverse_transform(y_test_lag.reshape(-1, 1)).ravel()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

print(f"\n── Sequential GBR Results ──────────────")
print(f"  RMSE : {rmse:.2f} Wh")
print(f"  MAE  : {mae:.2f} Wh")
print(f"  MAPE : {mape:.2f}%")
print(f"  R²   : {r2:.4f}")

# ── Save ──────────────────────────────────────────────────────
with open('models/seq_model.pkl', 'wb') as f:
    pickle.dump(seq_model, f)

print("\n Saved → models/seq_model.pkl")
