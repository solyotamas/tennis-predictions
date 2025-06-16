import sys
import os
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from player import Tennis_Player


df_2010_2024 = pd.read_csv("datasets/cleaned_atp_matches_2010_2024.csv")
df_2000_2024 = pd.read_csv("datasets/cleaned_atp_matches_2000_2024.csv")

# =========================

def init_player_dict(df: pd.DataFrame):
    unique_names = set(df["winner_name"]).union(set(df["loser_name"]))

    players = {}

    for name in unique_names:
        player = Tennis_Player(name)
        player.elo = 1500
        player.surface_elo = {
            'Hard': 1500,
            'Clay': 1500,
            'Grass': 1500
        }
        players[name] = player.to_dict()

    return players


# =========================

players_2010_2024 = init_player_dict(df_2010_2024)
players_2000_2024 = init_player_dict(df_2000_2024)

with open("players_elos_from_2010.json", "w") as f:
    json.dump(players_2010_2024, f, indent=4, ensure_ascii=False)

with open("players_elos_from_2000.json", "w") as f:
    json.dump(players_2000_2024, f, indent=4, ensure_ascii=False)


