# =========================================================
# preprocess.py
# Advanced Preprocessing Pipeline
# Python 3.13 Compatible
# =========================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
def load_data(path):

    try:

        df = pd.read_csv(path)

        print(" Dataset Loaded Successfully")
        print(f" Dataset Shape: {df.shape}")

        return df

    except Exception as e:

        print(" Error Loading Dataset:", e)
        return None


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------
def clean_data(df):

    print("\n Cleaning Data...")

    # -------------------------------------------------
    # Convert Column Names to Lowercase
    # -------------------------------------------------
    df.columns = df.columns.str.lower()

    # -------------------------------------------------
    # Remove Duplicate Rows
    # -------------------------------------------------
    before = len(df)

    df.drop_duplicates(inplace=True)

    after = len(df)

    print(f" Removed {before - after} duplicate rows")

    # -------------------------------------------------
    # REMOVE INFINITE VALUES
    # -------------------------------------------------
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    print(" Infinity values handled")

    # -------------------------------------------------
    # DATE COLUMN HANDLING
    # -------------------------------------------------
    possible_date_cols = [
        'order_date',
        'ship_date',
        'order date',
        'ship date'
    ]

    for col in possible_date_cols:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors='coerce'
            )

            # Fill missing dates
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()

    # -------------------------------------------------
    # NUMERIC COLUMN HANDLING
    # -------------------------------------------------
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    print(f" Numeric Columns: {len(numeric_cols)}")

    # Replace very large values
    for col in numeric_cols:

        try:

            df[col] = np.where(
                df[col] > 1e15,
                np.nan,
                df[col]
            )

            df[col] = np.where(
                df[col] < -1e15,
                np.nan,
                df[col]
            )

        except:
            pass

    # Fill numeric missing values
    if len(numeric_cols) > 0:

        num_imputer = SimpleImputer(
            strategy='median'
        )

        df[numeric_cols] = num_imputer.fit_transform(
            df[numeric_cols]
        )

    print(" Numeric missing values handled")

    # -------------------------------------------------
    # CATEGORICAL COLUMN HANDLING
    # -------------------------------------------------
    categorical_cols = df.select_dtypes(
        include=['object', 'string']
    ).columns

    print(f" Categorical Columns: {len(categorical_cols)}")

    if len(categorical_cols) > 0:

        cat_imputer = SimpleImputer(
            strategy='most_frequent'
        )

        df[categorical_cols] = cat_imputer.fit_transform(
            df[categorical_cols]
        )

    print(" Categorical missing values handled")

    # -------------------------------------------------
    # REMOVE USELESS COLUMNS
    # -------------------------------------------------
    useless_cols = [
        'unnamed: 0',
        'index',
        'column'
    ]

    for col in useless_cols:

        if col in df.columns:

            df.drop(columns=col, inplace=True)

    # -------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------
    order_col = None
    ship_col = None

    for col in df.columns:

        if 'order' in col and 'date' in col:
            order_col = col

        if 'ship' in col and 'date' in col:
            ship_col = col

    # Shipping Days
    if order_col and ship_col:

        df['shipping_days'] = (
            df[ship_col] - df[order_col]
        ).dt.days

    # Date Features
    if order_col:

        df['order_year'] = df[order_col].dt.year
        df['order_month'] = df[order_col].dt.month
        df['order_day'] = df[order_col].dt.day
        df['order_weekday'] = df[order_col].dt.weekday

        df['is_weekend'] = df[
            'order_weekday'
        ].isin([5, 6]).astype(int)

    print(" Feature Engineering Completed")

    # -------------------------------------------------
    # REMOVE NEGATIVE VALUES (only for columns where negatives are errors)
    # -------------------------------------------------
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    negative_fix_cols = ['sales', 'quantity']

    for col in negative_fix_cols:

        if col in df.columns:

            df[col] = np.where(
                df[col] < 0,
                0,
                df[col]
            )

    print(" Negative values handled (sales/quantity only; profit kept as-is)")

    # -------------------------------------------------
    # OUTLIER HANDLING (sales excluded to keep large orders intact)
    # -------------------------------------------------
    outlier_cols = [c for c in numeric_cols if c != 'sales']

    for col in outlier_cols:

        try:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - (1.5 * IQR)
            upper = Q3 + (1.5 * IQR)

            df[col] = np.clip(
                df[col],
                lower,
                upper
            )

        except:
            pass

    print(" Outliers handled")

    print(" Data Cleaning Completed")

    return df


# ---------------------------------------------------------
# ENCODE CATEGORICAL DATA
# ---------------------------------------------------------
def encode_data(df):

    print("\n Encoding Data...")

    label_encoders = {}

    categorical_cols = df.select_dtypes(
        include=['object', 'string']
    ).columns

    for col in categorical_cols:

        try:

            le = LabelEncoder()

            df[col] = le.fit_transform(
                df[col].astype(str)
            )

            label_encoders[col] = le

        except Exception as e:

            print(f" Encoding Error in {col}: {e}")

    print(" Encoding Completed")

    return df, label_encoders


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------
def preprocess_pipeline(path):

    df = load_data(path)

    if df is not None:

        df = clean_data(df)

        df, encoders = encode_data(df)

        print("\n Final Dataset Shape:", df.shape)

        return df, encoders

    return None, None


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    # Dataset Path
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "superstore_cleaned_complete.csv")

    # Run Pipeline
    processed_df, encoders = preprocess_pipeline(
        data_path
    )

    # Show Data
    print("\n First 5 Rows:\n")

    print(processed_df.head())

    # Save Processed Dataset
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_superstore.csv")

    processed_df.to_csv(
        output_path,
        index=False
    )

    print("\n Processed Dataset Saved Successfully")
    print(f" File Saved At: {output_path}")