
import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and feature list
@st.cache_resource # Caches the model so it doesn't reload on every button click
def load_resources():
    rf_model = joblib.load("rf_steam_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return rf_model, model_features

model, features = load_resources()

# 2. Build the Web App UI
st.title("🎮 Steam Game Sales Predictor")
st.markdown("Welcome! Enter the game's details and tags to predict its commercial success on Steam.")

st.sidebar.header("Game Attributes")

# Numeric inputs
release_year = st.sidebar.number_input("Release Year", min_value=2000, max_value=2030, value=2024, step=1)
positive_ratings = st.sidebar.number_input("Positive Ratings", min_value=0, value=500, step=100)
negative_ratings = st.sidebar.number_input("Negative Ratings", min_value=0, value=100, step=10)
average_playtime = st.sidebar.number_input("Average Playtime (minutes)", min_value=0, value=120, step=10)

# Tag selection
# Filter out the numeric columns so the user only sees the 0/1 tags in the dropdown
tag_options = [f for f in features if f not in ['release_year', 'positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime']]
selected_tags = st.sidebar.multiselect("Select Game Tags", options=tag_options, default=['action', 'indie', 'singleplayer'])

# 3. Prediction Logic
if st.button("Predict Sales Tier 🚀"):
    # Create a dictionary with all features initialized to 0
    input_data = {feature: 0 for feature in features}
    
    # Update numeric features
    input_data['release_year'] = release_year
    input_data['positive_ratings'] = positive_ratings
    input_data['negative_ratings'] = negative_ratings
    input_data['average_playtime'] = average_playtime
    # We estimate median playtime as roughly 80% of average playtime for simplicity
    input_data['median_playtime'] = int(average_playtime * 0.8)
    
    # Update tag features (set selected tags to 1)
    for tag in selected_tags:
        if tag in input_data:
            input_data[tag] = 1
            
    # Convert to DataFrame (model expects a 2D array/DataFrame)
    input_df = pd.DataFrame([input_data])
    
    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    # Display the result
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction == 0:
        st.error("📉 **Tier 0: Flop** (Expected Sales < 20,000 copies). \n\n*The model suggests this game might struggle to find a large audience.*")
    elif prediction == 1:
        st.warning("📊 **Tier 1: Average** (Expected Sales 20,000 - 200,000 copies). \n\n*Solid performance! This indicates a healthy standard release.*")
    else:
        st.success("🏆 **Tier 2: Hit!** (Expected Sales > 200,000 copies). \n\n*Congratulations! This game has the DNA of a Steam blockbuster!*")