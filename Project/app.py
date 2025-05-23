import streamlit as st
import pandas as pd
from model_utils import train_model, predict

st.set_page_config(page_title="IMDB Rating Predictor", layout="centered")

st.title("🎬 Predict IMDB Movie Ratings")
st.write("Upload a CSV file with movie features (including `ratings` for training) or without for prediction.")

# Upload file
file = st.file_uploader("Upload your CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Preview of Data")
    st.write(df.head(1000))

    # Check if we’re training or predicting
    if 'Rating_from_10' in df.columns:
        st.subheader("Training Models...")
        try:
            model, encoders, features, rf_rmse, lr_rmse = train_model(file)
            st.success("Models trained successfully!")
            st.write(f"📈 **Random Forest RMSE:** {rf_rmse:.2f}")
            st.write(f"📉 **Linear Regression RMSE:** {lr_rmse:.2f}")

            st.session_state['model'] = model
            st.session_state['encoders'] = encoders
            st.session_state['features'] = features


        except Exception as e:

            st.error(f"Error during training: {e}")


    elif 'model' in st.session_state:
        st.subheader("Predicting Ratings from CSV...")
        try:
            preds = predict(st.session_state['model'], st.session_state['encoders'], df, st.session_state['features'])
            df['Predicted IMDB Rating'] = preds
            st.write(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predicted_ratings.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please first upload a dataset containing 'Rating_from_10' to train the model.")

    # Manual Input Prediction
    if 'model' in st.session_state:
        st.subheader("🎯 Manual Movie Input for Prediction")

        user_input = {
            'Movie_name': st.text_input("Movie Name"),
            'Year': st.number_input("Release Year", min_value=1900, max_value=2100, value=2020),
            'Runtime_in_min': st.number_input("Runtime (minutes)", min_value=1, value=90),
            'Genre': st.text_input("Genre"),
            'Gross_in_$_M': st.number_input("Gross Budget ($)", min_value=0.0, value=00.0)
        }

        if st.button("Predict Rating"):
            try:
                input_df = pd.DataFrame([user_input])
                preds = predict(st.session_state['model'], st.session_state['encoders'], input_df,
                                st.session_state['features'])
                st.success(f"⭐ Predicted IMDB Rating: {preds[0]:.2f}")

            except Exception as e:
                st.error(f"Manual prediction failed: {e}")
