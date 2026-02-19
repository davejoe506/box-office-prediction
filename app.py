import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and feature columns
MODEL_PATH = "model.pkl"
FEATURES_PATH = "model_features.pkl"

st.set_page_config(page_title="Box Office Predictor", page_icon="🎬", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)

try:
    model, feature_cols = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Did you run 05_modeling.py first?")
    st.stop()

st.title("🎬 Movie Box Office Predictor")
st.markdown("Enter the details of a hypothetical movie to see how much money our model predicts it will make globally!")

# UI Layout
col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("Budget (Millions, US Dollars)", min_value=1, max_value=500, value=100)
    runtime = st.number_input("Runtime (Minutes)", min_value=60, max_value=240, value=120)
    season = st.selectbox(
        "Release Season", 
        ["Summer Blockbuster", "Holiday Season", "Spring Fall", "Dump Months"],
        help="""
        **How months are categorized:**
        * **Summer Blockbuster:** May, June, July
        * **Holiday Season:** November, December
        * **Spring/Fall:** March, April, October
        * **Dump Months:** January, February, August, September
        """
    )
    is_franchise = st.checkbox("Is this part of a franchise?")

with col2:
    director_score = st.number_input("Director's Past Box Office (Millions, US Dollars)", min_value=0, max_value=5000, value=250)
    actor_score = st.number_input("Top Actor's Past Box Office (Millions, US Dollars)", min_value=0, max_value=5000, value=300)
    primary_genre = st.selectbox("Primary Genre", ["Action", "Adventure", "Animation", "Comedy", "Drama", "Science Fiction", "Horror", "Thriller"])

# Processing input
if st.button("Predict Box Office Revenue 💸"):
    with st.spinner('Calculating Hollywood Magic...'):
        
        # 1. Create dictionary of all zeros for features
        input_data = {col: 0 for col in feature_cols}
        
        # 2. Populate numerical values
        input_data['budget_adj'] = budget * 1_000_000
        input_data['runtime'] = runtime
        input_data['is_franchise'] = 1 if is_franchise else 0
        input_data['director_score'] = director_score
        input_data['actor_score'] = actor_score
        
        # 3. Populate one-hot encoded categorical values
        season_col = f"season_{season}"
        if season_col in input_data:
            input_data[season_col] = 1
            
        genre_col = f"genre_{primary_genre}"
        if genre_col in input_data:
            input_data[genre_col] = 1
            
        # 4. Convert to DataFrame and predict
        input_df = pd.DataFrame([input_data])
        
        #  Model predicts the log of revenue, so it must be exponentiated
        log_pred = model.predict(input_df)[0]
        actual_dollars = np.expm1(log_pred)
        
        # Output
        st.success("Prediction Complete!")
        
        # Format output
        if actual_dollars >= 1_000_000_000:
            st.metric(label="Predicted Global Revenue", value=f"${actual_dollars / 1_000_000_000:.2f} Billion")
            st.balloons()
        else:
            st.metric(label="Predicted Global Revenue", value=f"${actual_dollars / 1_000_000:.1f} Million")

st.markdown("---")
st.caption("Model built with XGBoost. Predictions are based on historical data adjusted for 2024 inflation.")