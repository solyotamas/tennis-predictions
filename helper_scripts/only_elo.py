import pandas as pd


df = pd.read_csv("datasets/elo_driven_features_2010_2024.csv")


minimal_features = [
    'player1_welo', 'player2_welo',
    'player1_surface_welo', 'player2_surface_welo',
    'player1_won'
]

df_minimal = df[minimal_features]


df_minimal.to_csv("datasets/minimal_elo_dataset.csv", index=False)