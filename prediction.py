import pandas as pd
import joblib
import numpy as np
import sys, os
import json
import streamlit as st
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from player_stats import Player

# import your custom feature functions
from feature_computing.elo_driven_features import (
    recent_win_percent_exponential_decay,
    recent_surface_win_percent_exponential_decay
)

# =========================

def encode_categorical_features(X):
    categorical_cols = ['player1_hand', 'player2_hand', 'surface', 'tourney_level', 'best_of']
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    X = pd.get_dummies(X, columns=categorical_cols)

    round_order = {
        'R128': 1,
        'R64': 2,
        'R32': 3,
        'R16': 4,
        'QF': 5,
        'SF': 6,
        'F': 7
    }

    X['tourney_round'] = X['tourney_round'].map(round_order)

    return X

def align_columns(new_sample, final_columns):
    # Add missing columns
    for col in final_columns:
        if col not in new_sample.columns:
            new_sample[col] = 0
    # Drop unexpected columns
    new_sample = new_sample[final_columns]
    return new_sample

def build_features_full(player1, player2, surface, current_date, tourney_level, tourney_round, best_of):
    features = {
        'player1_welo': player1.elo,
        'player2_welo': player2.elo,
        'player1_surface_welo': player1.surface_elo[surface],
        'player2_surface_welo': player2.surface_elo[surface],
        'player1_rank': player1.rank,
        'player2_rank': player2.rank,
        'player1_rank_points': player1.rank_points,
        'player2_rank_points': player2.rank_points,

        'player1_recent_10_exponential_decay': recent_win_percent_exponential_decay(player1, current_date, window=10), 
        'player1_recent_25_exponential_decay': recent_win_percent_exponential_decay(player1, current_date, window=25), 
        'player1_recent_50_exponential_decay': recent_win_percent_exponential_decay(player1, current_date, window=50),
        'player2_recent_10_exponential_decay': recent_win_percent_exponential_decay(player2, current_date, window=10), 
        'player2_recent_25_exponential_decay': recent_win_percent_exponential_decay(player2, current_date, window=25), 
        'player2_recent_50_exponential_decay': recent_win_percent_exponential_decay(player2, current_date, window=50),

        'player1_recent_10_surface_exponential_decay': recent_surface_win_percent_exponential_decay(player1, surface, current_date, window=10),
        'player1_recent_25_surface_exponential_decay': recent_surface_win_percent_exponential_decay(player1, surface, current_date, window=25),
        'player1_recent_50_surface_exponential_decay': recent_surface_win_percent_exponential_decay(player1, surface, current_date, window=50),
        'player2_recent_10_surface_exponential_decay': recent_surface_win_percent_exponential_decay(player2, surface, current_date, window=10),
        'player2_recent_25_surface_exponential_decay': recent_surface_win_percent_exponential_decay(player2, surface, current_date, window=25),
        'player2_recent_50_surface_exponential_decay': recent_surface_win_percent_exponential_decay(player2, surface, current_date, window=50),

        'player1_age': player1.age,
        'player2_age': player2.age,
        'player1_height': player1.height,
        'player2_height': player2.height,
        'player1_hand': player1.hand,
        'player2_hand': player2.hand,

        'surface': surface,
        'tourney_level': tourney_level,
        'tourney_round': tourney_round,
        'best_of': best_of
    }
    return features

def build_features_minimal(player1, player2, surface):
    features = {
        'player1_welo': player1.elo,
        'player2_welo': player2.elo,
        'player1_surface_welo': player1.surface_elo[surface],
        'player2_surface_welo': player2.surface_elo[surface],
    }
    return features

def predict_match_streamlit(players, full_model, full_columns, minimal_model, minimal_columns):
    
    st.title("ATP Tennis Match Predictor")
    st.info("Using data until end of 2024")

    player_list = sorted(players.keys())

    default_player1 = player_list.index("Holger Rune") if "Holger Rune" in player_list else 0
    default_player2 = player_list.index("Novak Djokovic") if "Novak Djokovic" in player_list else 0

    with st.form("match_form"):
        player1_name = st.selectbox("Select Player 1", player_list, index=default_player1)
        player2_name = st.selectbox("Select Player 2", player_list, index=default_player2)
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
        tourney_level = st.selectbox("Tournament Level", ["G", "M", "A"])
        tourney_round = st.selectbox("Tournament Round", ["R128", "R64", "R32", "R16", "QF", "SF", "F"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        current_date = 20241231
        best_of = 5 if tourney_level == "G" else 3

        player1 = players[player1_name]
        player2 = players[player2_name]

        #### FULL MODEL
        features_full = build_features_full(player1, player2, surface, current_date, tourney_level, tourney_round, best_of)
        df_full = pd.DataFrame([features_full])
        df_full = encode_categorical_features(df_full)
        df_full = align_columns(df_full, full_columns)
        prob_full_p1 = full_model.predict_proba(df_full)[0][1]
        prob_full_p2 = 1 - prob_full_p1

        st.markdown("---")
        st.subheader("Full Model Prediction")

        col1, col2 = st.columns(2)
        col1.metric(label=f"**{player1_name} win probability**", value=f"{prob_full_p1:.2%}")
        col2.metric(label=f"**{player2_name} win probability**", value=f"{prob_full_p2:.2%}")

        #### MINIMAL MODEL
        features_min = build_features_minimal(player1, player2, surface)
        df_min = pd.DataFrame([features_min])
        df_min = align_columns(df_min, minimal_columns)
        prob_min_p1 = minimal_model.predict_proba(df_min)[0][1]
        prob_min_p2 = 1 - prob_min_p1

        st.markdown("---")
        st.subheader("Minimal Model Prediction")

        col1, col2 = st.columns(2)
        col1.metric(label=f"**{player1_name} win probability**", value=f"{prob_min_p1:.2%}")
        col2.metric(label=f"**{player2_name} win probability**", value=f"{prob_min_p2:.2%}")

        st.markdown("---")
        st.success("Prediction completed successfully!")


def predict_match_cli(player1, player2, surface, current_date, tourney_level, tourney_round, best_of, full_model, full_columns, minimal_model, minimal_columns):
    
    features_full = build_features_full(player1, player2, surface, current_date, tourney_level, tourney_round, best_of)
    df_full = pd.DataFrame([features_full])
    df_full = encode_categorical_features(df_full)
    df_full = align_columns(df_full, full_columns)

    prob_full_p1 = full_model.predict_proba(df_full)[0][1]
    prob_full_p2 = 1 - prob_full_p1
    print("FULL MODEL:")
    print(f"Win probability for {player1.name} is {prob_full_p1:.2%}")
    print(f"Win probability for {player2.name} is {prob_full_p2:.2%}")

    # Minimal model
    features_min = build_features_minimal(player1, player2, surface)
    df_min = pd.DataFrame([features_min])
    df_min = align_columns(df_min, minimal_columns)
    prob_min_p1 = minimal_model.predict_proba(df_min)[0][1]
    prob_min_p2 = 1 - prob_min_p1
    print("MINIMAL MODEL:")
    print(f"Win probability for {player1.name} is {prob_min_p1:.2%}")
    print(f"Win probability for {player2.name} is {prob_min_p2:.2%}")

# =========================

full_model = joblib.load("models/elo_with_complimentary_model.pkl")
full_columns = joblib.load("models/elo_with_complimentary_columns.pkl")

minimal_model = joblib.load("models/elo_only_model.pkl")
minimal_columns = joblib.load("models/elo_only_columns.pkl")

# Load players
with open("players_output_2010_2024.json", "r") as f:
    player_data = json.load(f)

players = {name: Player.from_dict(data) for name, data in player_data.items()}

# ============== what to predict if using cli

# Select players
player1 = players["Holger Rune"]
player2 = players["Novak Djokovic"]

surface = "Clay"
current_date = 20241230
tourney_level = "M"
tourney_round = "QF"
best_of = 5 if tourney_level == "G" else 3

# ========================= CLI version

#predict_match_cli(player1, player2, surface, current_date, tourney_level, tourney_round, best_of, full_model, full_columns, minimal_model, minimal_columns)

# ========================= UI version

# py -m streamlit run prediction.py
predict_match_streamlit(players, full_model, full_columns, minimal_model, minimal_columns)
