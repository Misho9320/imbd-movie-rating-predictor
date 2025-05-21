import streamlit as st
import pandas as pd
from model_utils import train_model, predict

st.set_page_config(page_title="IMDB Rating Predictor", layout="centered")

st.title("🎬 Predict IMDB Movie Ratings")
st.write("Upload a CSV file with movie features (including `imdb_rating` for training) or without for prediction.")

# Upload file
file = st.file_uploader("Upload your CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Preview of Data")
    st.write(df.head(1000))

    # Check if we’re training or predicting
    if 'Rating_from_10' in df.columns:
        st.subheader("Training Model...")
        try:
            model, encoders, features, rmse = train_model(file)
            st.success(f"Model trained successfully! RMSE: {rmse:.2f}")
            st.session_state['model'] = model
            st.session_state['encoders'] = encoders
            st.session_state['features'] = features

        except Exception as e:
            st.error(f"Error during training: {e}")
    elif 'model' in st.session_state:
        st.subheader("Predicting Ratings...")
        try:
            preds = predict(st.session_state['model'], st.session_state['encoders'], df, st.session_state['features'])
            df['Predicted IMDB Rating'] = preds
            st.write(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predicted_ratings.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please first upload a dataset containing 'imdb_rating' to train the model.")
