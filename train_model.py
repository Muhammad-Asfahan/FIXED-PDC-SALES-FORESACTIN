# =========================================================
# train_model.py
# Sales Demand Prediction Training
# =========================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print(" Loading dataset...")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_superstore.csv")
df = pd.read_csv(DATA_PATH)

print(" Dataset Loaded Successfully")
print(f" Dataset Shape: {df.shape}")

# ---------------------------------------------------------
# REMOVE DATE COLUMNS
# ---------------------------------------------------------

date_columns = []

for col in df.columns:

    if "date" in col.lower():

        date_columns.append(col)

if len(date_columns) > 0:

    print("\n Removing Date Columns:")
    print(date_columns)

    df.drop(columns=date_columns, inplace=True)

# ---------------------------------------------------------
# REMOVE REMAINING STRING COLUMNS
# ---------------------------------------------------------

string_cols = df.select_dtypes(
    include=['object', 'string']
).columns

if len(string_cols) > 0:

    print("\n Removing String Columns:")
    print(list(string_cols))

    df.drop(columns=string_cols, inplace=True)

# ---------------------------------------------------------
# TARGET COLUMN
# ---------------------------------------------------------

TARGET_COLUMN = "sales"

if TARGET_COLUMN not in df.columns:

    raise Exception(
        f" Target column '{TARGET_COLUMN}' not found!"
    )

# ---------------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------------

leaking_cols = ['profit', 'profit_margin', 'sales_per_unit', 'shipping_cost', 'shipping_days', 'order_id', 'customer_id', 'customer_name', 'product_id']
X = df.drop(columns=[TARGET_COLUMN] + leaking_cols, errors='ignore')
y = df[TARGET_COLUMN]

print("\n Features Prepared")

print(f" X Shape: {X.shape}")
print(f" y Shape: {y.shape}")

# ---------------------------------------------------------
# FINAL SAFETY CHECK
# ---------------------------------------------------------

print("\n Checking Data Types...\n")

print(X.dtypes)

# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

train_mask = df["order_year"] <= 2013
test_mask = df["order_year"] == 2014

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print("\n Train-Test Split Completed")

print(f" Training Samples: {X_train.shape[0]}")
print(f" Testing Samples: {X_test.shape[0]}")

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

print("\n Training Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print(" Model Training Completed")

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

print("\n Making Predictions...")

y_pred = model.predict(X_test)

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Results\n")

print(f" MAE       : {mae:.2f}")
print(f" MSE       : {mse:.2f}")
print(f" RMSE      : {rmse:.2f}")
print(f" R2 Score  : {r2:.4f}")

baseline_pred = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print("\n Baseline (predicting average sales every time)")
print(f" Baseline MAE : {baseline_mae:.2f}")
print(f" Baseline R2  : {baseline_r2:.4f}")

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

lin_mae = mean_absolute_error(y_test, lin_pred)
lin_r2 = r2_score(y_test, lin_pred)

print("\n Linear Regression (comparison model)")
print(f" Linear Regression MAE : {lin_mae:.2f}")
print(f" Linear Regression R2  : {lin_r2:.4f}")

# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

print("\n Top 10 Important Features\n")

importance_df = pd.DataFrame({

    'Feature': X.columns,
    'Importance': model.feature_importances_

})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print(importance_df.head(10))

# ---------------------------------------------------------
# SAVE MODEL + FEATURE COLUMNS
# ---------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_superstore.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sales_prediction_model.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "model_features.pkl"
)

# Save Model
joblib.dump(model, MODEL_PATH)

# Save Feature Columns
joblib.dump(X.columns.tolist(), FEATURES_PATH)

print("\n Model Saved Successfully")
print(f" Model Path: {MODEL_PATH}")

print("\n Feature Columns Saved Successfully")
print(f" Features Path: {FEATURES_PATH}")