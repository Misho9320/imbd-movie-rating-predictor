import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Train and save model
def train_model(csv_file):
    try:
        df = pd.read_csv(
            '/Project/imdb.csv',
            encoding='utf-8', on_bad_lines='skip'
        )
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    # Select only required features + target
    selected_features = ['Movie_name', 'Year', 'Runtime_in_min', 'Genre', 'Gross_in_$_M', 'Rating_from_10']
    df = df[selected_features]

    # Clean Year column (extract valid 4-digit years)
    df['Year'] = df['Year'].astype(str).apply(lambda x: int(re.search(r'\d{4}', x).group()) if re.search(r'\d{4}', x) else np.nan)

    # Convert numeric columns safely
    for col in ['Year', 'Runtime_in_min', 'Gross_in_$_M', 'Rating_from_10']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)  # Drop rows with missing or invalid values

    # Encode categorical features
    label_encoders = {}
    for col in ['Movie_name', 'Genre']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    if 'Rating_from_10' not in df.columns:
        raise ValueError("Column 'Rating_from_10' is required in the dataset.")

    X = df.drop('Rating_from_10', axis=1)
    y = df['Rating_from_10']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_rmse = np.sqrt(rf_mse)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_preds)
    lr_rmse = np.sqrt(lr_mse)

    return rf_model, label_encoders, X.columns.tolist(), rf_rmse, lr_rmse

# Predict using trained model
def predict(model, label_encoders, input_df, feature_names):
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    input_df = input_df[feature_names]
    predictions = model.predict(input_df)
    return predictions
