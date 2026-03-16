import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and feature list
@st.cache_resource 
def load_resources():
    # Loading the model and the exact feature list used during training
    rf_model = joblib.load("rf_steam_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return rf_model, model_features

model, features = load_resources()

# 2. Build the Web App UI
st.set_page_config(page_title="Steam Sales Predictor", page_icon="🎮")
st.title("🎮 Steam Game Sales Predictor")
st.markdown("Enter the game's details and tags below to predict its commercial success tier on Steam.")

st.sidebar.header("Game Attributes")

# Numeric inputs - including all missing features from model_features.pkl
release_year = st.sidebar.number_input("Release Year", min_value=1980, max_value=2030, value=2024)
price = st.sidebar.number_input("Price (USD)", min_value=0.0, max_value=200.0, value=9.99, step=1.0)
required_age = st.sidebar.number_input("Required Age", min_value=0, max_value=18, value=0)
achievements = st.sidebar.number_input("Number of Achievements", min_value=0, max_value=5000, value=20)
positive_ratings = st.sidebar.number_input("Positive Ratings", min_value=0, value=500, step=100)
negative_ratings = st.sidebar.number_input("Negative Ratings", min_value=0, value=100, step=10)
average_playtime = st.sidebar.number_input("Average Playtime (minutes)", min_value=0, value=120, step=10)

# Tag selection
# Filter out numeric columns so only 0/1 binary tags appear in the multiselect
numeric_cols = ['release_year', 'positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime', 'price', 'achievements', 'required_age']
tag_options = [f for f in features if f not in numeric_cols]
selected_tags = st.sidebar.multiselect("Select Game Tags", options=tag_options, default=['action', 'indie', 'singleplayer'])

# 3. Prediction Logic
if st.button("Predict Sales Tier 🚀"):
    # Initialize a dictionary with all features from model_features.pkl set to 0
    input_data = {feature: 0 for feature in features}
    
    # Map the UI inputs to the dictionary
    input_data['release_year'] = release_year
    input_data['price'] = price
    input_data['required_age'] = required_age
    input_data['achievements'] = achievements
    input_data['positive_ratings'] = positive_ratings
    input_data['negative_ratings'] = negative_ratings
    input_data['average_playtime'] = average_playtime
    # Simple estimation for median playtime
    input_data['median_playtime'] = int(average_playtime * 0.8)
    
    # Set selected tags to 1
    for tag in selected_tags:
        if tag in input_data:
            input_data[tag] = 1
            
    # CRITICAL: Convert to DataFrame and reorder columns to match 'features' list exactly
    input_df = pd.DataFrame([input_data])[features]
    
    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    # Display the result
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction == 0:
        st.error("📉 **Tier 0: Flop** (Expected Sales < 20,000 copies). \n\nThe model suggests this game might struggle to find a large audience.")
    elif prediction == 1:
        st.warning("📊 **Tier 1: Average** (Expected Sales 20,000 - 200,000 copies). \n\nSolid performance! This indicates a healthy standard release.")
    else:
        st.success("🏆 **Tier 2: Hit!** (Expected Sales > 200,000 copies). \n\nCongratulations! This game has the DNA of a Steam blockbuster!")

    # Debug Section
    with st.expander("Show Model Input Data (Debug)"):
        st.write(input_df)