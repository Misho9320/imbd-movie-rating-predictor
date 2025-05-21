import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Train and save model
def train_model(csv_path):
    df = pd.read_csv('C:/Users/MISHO/Desktop/My Modules/Honours/Artificial Intelligence - BICT411 (1st)/Project/imdb.csv')
    st.write(f"loaded {len(df)} rows from dataset")

    # Basic preprocessing
    df = df.dropna()
    df = df.select_dtypes(include=['number', 'object'])  # drop datetime or unsupported types

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    if 'Rating_from_10' not in df.columns:
        raise ValueError("Column 'imdb_rating' is required in the dataset.")

    X = df.drop('Rating_from_10', axis=1)
    y = df['Rating_from_10']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, label_encoders, X.columns.tolist(), rmse

# Predict using trained model
def predict(model, label_encoders, input_df, feature_names):
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    input_df = input_df[feature_names]
    predictions = model.predict(input_df)
    return predictions
