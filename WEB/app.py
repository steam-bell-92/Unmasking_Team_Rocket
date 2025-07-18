import numpy as np
import streamlit as st
import pandas as pd
import joblib
import time
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Unmasking Team Rocket",
    page_icon="🚀",
)

st.write("# Welcome to my project! 👋")

model = joblib.load("WEB/xgb_model.pkl")
feature_columns = joblib.load("WEB/model_features.pkl")

st.markdown(
    "[GitHub Repo](https://github.com/steam-bell-92/Unmasking_Team_Rocket/blob/main/CODES/Unmasking_Team_Rocket.ipynb) 🔗"
)

with st.form('input_form'):
    st.write('Team Rocket Project')
    age = st.slider('Age', 1, 100)
    city = st.selectbox('City', [
        'Pewter City', 'Viridian City', 'Pallet Town', 'Cerulean City',
        'Lavender Town', 'Celadon City', 'Saffron City', 'Cinnabar Island',
        'Fuchsia City', 'Vermilion City'
    ])
    economy = st.selectbox('Economic Status', ['High', 'Middle', 'Low'])
    profession = st.selectbox('Profession', [
        'Fisherman', 'PokéMart Seller', 'Police Officer', 'Gym Leader Assistant',
        'Daycare Worker', 'Casino Worker', 'Rocket Grunt', 'Breeder', 'Nurse',
        'Researcher', 'Elite Trainer', 'Scientist', 'Black Market Dealer',
        'Champion', 'Biker', 'Underground Battler'
    ])
    pokemon_type = st.selectbox('Most Used Pokemon Type', [
        'Rock', 'Grass', 'Poison', 'Dragon', 'Ground', 'Ghost', 'Bug',
        'Fighting', 'Electric', 'Flying', 'Ice', 'Psychic', 'Fire',
        'Fairy', 'Water', 'Dark', 'Steel', 'Normal'
    ])
    avg_level = st.slider('Average Level of Pokemon', 1, 100)
    pokeball = st.selectbox('PokéBall Usage', [
        'DuskBall', 'HealBall', 'NetBall', 'UltraBall', 'TimerBall',
        'MasterBall', 'LuxuryBall', 'DarkBall', 'PokéBall', 'GreatBall'
    ])
    win_ratio = st.slider('Win Ratio', 0, 100)
    strategy = st.selectbox('Battle Strategy', ['Aggressive', 'Unpredictable', 'Defensive'])
    migration = st.slider('Number of Migration', 0, 100)
    item = st.selectbox('Rare Item Holder', [False, True])

    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        'Age': age,
        "City": city,
        'Economic Status': economy,
        'Profession': profession,
        'Most Used Pokemon Type': pokemon_type,
        'Average Level of Pokemon': avg_level,
        'PokéBall Usage': pokeball,
        'Win Ratio': win_ratio,
        'Battle Strategy': strategy,
        'Number of Migration': migration,
        'Rare Item Holder': item,
    }

    input_df = pd.DataFrame([input_dict])
    input_df_num = input_df[['Age', 'Average Level of Pokemon', 'Win Ratio', 'Number of Migration', 'Rare Item Holder']]
    input_df_cat = input_df.drop(columns=input_df_num.columns)      # Dropping numerical columns to get categorial columns
    input_encoded_cat = pd.get_dummies(input_df_cat)                # Done to take in account the encoding of categorial columns
    input_encoded = pd.concat([input_df_num, input_encoded_cat], axis=1)

    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    proba = model.predict_proba(input_encoded)[0][1]
    pred = int(proba >= 0.45)                               # Same threshold as in model, used for classfying

    st.subheader("🔍 Prediction Result")
    st.caption("Model threshold: 0.45 (slightly aggressive to catch Team Rocket members)")

    def progress(percent, member=True):
        percent = min(100, percent)
        gradient = (
            "linear-gradient(to right, #ff0000, #ffa500, #ffff00)"      # Making gradient on progress bar
            if member else
            "linear-gradient(to right, #00ff00, #66ffcc, #66ccff)"      # Making gradient on progress bar
        )
        progress_placeholder = st.empty()
        for per in range(1, int(percent) + 1, 2): 
            html = f"""
                <div style="background-color: #262730; border-radius: 8px; width: 100%; height: 15px; position: relative; margin-bottom: 8px;">
                    <div style="background: {gradient}; width: {per:.2f}%; height: 100%; border-radius: 8px;"></div>
                </div>
                <p style="text-align: center; font-size: 16px; margin-top: -12px;">
                    Confidence: {per:.2f}%
                </p>
            """
            progress_placeholder.markdown(html, unsafe_allow_html=True)
            time.sleep(0.05)

    if pred:
        progress(proba * 100, member=True)                  # Progress bar for Team Rocket memeber (proba)
        st.warning("🚨 Likely Team Rocket Member!")
    else:
        progress((1 - proba) * 100, member=False)           # Progress bar for Non - Team Rocket Member (1-proba)
        st.success("✅ Safe Trainer Detected.")