import numpy as np
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Unmasking Team Rocket",
    page_icon="Rocket",
)

st.write("# Welcome to my project! 👋")

model = joblib.load("xgb_model.pkl")
feature_columns = joblib.load("model_features.pkl")

st.markdown(
    "[GitHub Repo 🔗](https://github.com/steam-bell-92/Unmasking_Team_Rocket/blob/main/CODES/Unmasking_Team_Rocket.ipynb)"
)

with st.form('input_form'):
  st.write('Team Rocket Project')
  age = st.slider('Age', 1, 100)
  city = st.selectbox('City', ['Pewter City', 'Viridian City', 'Pallet Town', 'Cerulean City',
       'Lavender Town', 'Celadon City', 'Saffron City', 'Cinnabar Island',
       'Fuchsia City', 'Vermilion City'])
  economy = st.selectbox('Economic Staus', ['High', 'Middle', 'Low'])
  profession = st.selectbox('Profession', ['Fisherman', 'PokéMart Seller', 'Police Officer',
       'Gym Leader Assistant', 'Daycare Worker', 'Casino Worker',
       'Rocket Grunt', 'Breeder', 'Nurse', 'Researcher', 'Elite Trainer',
       'Scientist', 'Black Market Dealer', 'Champion', 'Biker',
       'Underground Battler'])
  pokemon_type = st.selectbox('Most Used Pokemon Type', ['Rock', 'Grass', 'Poison', 'Dragon', 'Ground', 'Ghost', 'Bug',
       'Fighting', 'Electric', 'Flying', 'Ice', 'Psychic', 'Fire',
       'Fairy', 'Water', 'Dark', 'Steel', 'Normal'])
  avg_level = st.slider('Average Level of Pokemon', 1, 100)
  pokeball = st.selectbox('PokéBall Usage', ['DuskBall', 'HealBall', 'NetBall', 'UltraBall', 'TimerBall',
       'MasterBall', 'LuxuryBall', 'DarkBall', 'PokéBall', 'GreatBall'])
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
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(input_encoded)[0][1]
    pred = int(proba >= 0.45)

    st.subheader("🔍 Prediction Result")
    st.progress(int(proba * 100))  # ✅ This line must be inside 'if submit'
    st.caption("Model threshold: 0.45 (slightly aggressive to catch Team Rocket members)")

    if pred:
        st.warning(f"🚨 Likely Team Rocket Member!\nConfidence: {proba*100:.2f}%")
    else:
        st.success(f"✅ Safe Trainer Detected.\nConfidence: {proba*100:.2f}%")