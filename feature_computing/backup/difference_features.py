import pandas as pd
import json
import random
import sys
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from player import Tennis_Player

# =========================

# Either 2000 or 2010
start_date = '2010' 

# If True, encode categorical features as one-hot encoded columns
# Player1 Hand, Player2 Hand, Surface, Tourney Level, Tourney Round, Best of
encode_categories = False

df = pd.read_csv(f"datasets/cleaned_atp_matches_{start_date}_2024.csv")
df = df.sort_values('tourney_date')

# =====================================

def compute_difference_features(df: pd.DataFrame, start_date: str, encode_categories: bool):

    dataset = []

    features = [
        # ELo, Surface Elo Diff
        'elo_diff', 'surface_elo_diff',

        # Rank, Rank Points Diff
        'rank_diff', 'rank_points_diff',

        # Age, Height, Hand Diff
        'age_diff', 'height_diff', 'hand_matchup',

        # Recent Win % Diff
        'recent_10_diff', 'recent_25_diff', 'recent_50_diff',
        'recent_10_diff_surface','recent_25_diff_surface', 'recent_50_diff_surface',

        # Head to Head Diff
        'h2h_diff',

        # Tournament Level, Round, Surface, Best of
        'tourney_level', 'tourney_round', 'surface', 'best_of',

        # Outcome
        'player1_won'
    ]     
            

    # ========================= Loading Players
    players_dict = {}
    if start_date == '2010':
        with open("players_elos_from_2010.json", "r") as f:
            players_dict = json.load(f)
    else:
        with open("players_elos_from_2000.json", "r") as f:
            players_dict = json.load(f)

    players = {name: Tennis_Player.from_dict(data) for name, data in players_dict.items()}

    # =========================

    for _, row in df.iterrows():
        winner_name = row['winner_name']
        loser_name = row['loser_name']
        surface = row['surface']
        tourney_date = row['tourney_date']
        

        # ========== Variety

        player1, player2 = None, None

        if random.random() < 0.5:
            player1 = players[winner_name]
            player2 = players[loser_name]
        else:
            player1 = players[loser_name]
            player2 = players[winner_name]
            
        

        # ============= Features

        # Elo and Surface Elo difference
        elo_diff = player1.elo - player2.elo
        surface_elo_diff = player1.surface_elo[surface] - player2.surface_elo[surface]


        # Rank and Rank Points difference
        player_1_rank = row['winner_rank'] if winner_name == player1.name else row['loser_rank']
        player_2_rank = row['winner_rank'] if winner_name == player2.name else row['loser_rank']
        rank_diff = player_1_rank - player_2_rank
        player_1_rank_points = row['winner_rank_points'] if winner_name == player1.name else row['loser_rank_points']
        player_2_rank_points = row['winner_rank_points'] if winner_name == player2.name else row['loser_rank_points']
        rank_points_diff = player_1_rank_points - player_2_rank_points

        # Age difference
        player_1_age = row['winner_age'] if winner_name == player1.name else row['loser_age']
        player_2_age = row['winner_age'] if winner_name == player2.name else row['loser_age']
        age_diff = player_1_age - player_2_age

        # Height difference
        player_1_height = row['winner_ht'] if winner_name == player1.name else row['loser_ht']
        player_2_height = row['winner_ht'] if winner_name == player2.name else row['loser_ht']
        height_diff = player_1_height - player_2_height

        # Hand matchup
        player_1_hand = row['winner_hand'] if winner_name == player1.name else row['loser_hand']
        player_2_hand = row['winner_hand'] if winner_name == player2.name else row['loser_hand']
        hand_matchup = f"{player_1_hand}_{player_2_hand}"


        # Recent win % difference, 10, 25, 50 matches
        player_1_recent_10 = player1.win_pct(matches = player1.recent_10, weight = 5)
        player_2_recent_10 = player2.win_pct(matches = player2.recent_10, weight = 5)
        recent_10_diff = player_1_recent_10 - player_2_recent_10
        player_1_recent_25 = player1.win_pct(matches = player1.recent_25, weight = 2)
        player_2_recent_25 = player2.win_pct(matches = player2.recent_25, weight = 2)
        recent_25_diff = player_1_recent_25 - player_2_recent_25
        player_1_recent_50 = player1.win_pct(matches = player1.recent_50, weight = 1)
        player_2_recent_50 = player2.win_pct(matches = player2.recent_50, weight = 1)
        recent_50_diff = player_1_recent_50 - player_2_recent_50

        # Recent win % surface difference, 10, 25, 50 matches
        player_1_recent_10_surface = player1.surface_win_pct(surface = surface, window = '10', weight = 5)
        player_2_recent_10_surface = player2.surface_win_pct(surface = surface, window = '10', weight = 5)
        recent_10_diff_surface = player_1_recent_10_surface - player_2_recent_10_surface
        player_1_recent_25_surface = player1.surface_win_pct(surface = surface, window = '25', weight = 2)
        player_2_recent_25_surface = player2.surface_win_pct(surface = surface, window = '25', weight = 2)
        recent_25_diff_surface = player_1_recent_25_surface - player_2_recent_25_surface
        player_1_recent_50_surface = player1.surface_win_pct(surface = surface, window = '50', weight = 1)
        player_2_recent_50_surface = player2.surface_win_pct(surface = surface, window = '50', weight = 1)
        recent_50_diff_surface = player_1_recent_50_surface - player_2_recent_50_surface


        # Head to Head diff
        h2h_diff = player1.get_h2h_diff(player2)

        # Tourney level
        tourney_level = row['tourney_level']

        # Round
        tourney_round = row['round']

        # Best of
        best_of = row['best_of']

        # Outcome
        player1_won = 1 if player1.name == winner_name else 0

        # ============================= 

        random_forest_ready_row = [
            elo_diff, surface_elo_diff,
            rank_diff, rank_points_diff,

            age_diff, height_diff, hand_matchup,

            recent_10_diff, recent_25_diff, recent_50_diff,
            recent_10_diff_surface, recent_25_diff_surface, recent_50_diff_surface,

            h2h_diff, 
            
            tourney_level, tourney_round, surface, best_of,
            player1_won
        ]

        dataset.append(random_forest_ready_row)



        # ============= Updating Players

        # recent matches
        player1.update_after_match(won = player1_won, surface = surface)
        player2.update_after_match(won = not player1_won, surface = surface)

        # elo and surface elo
        point_diff = abs(
            row['winner_set1'] + row['winner_set2'] + row['winner_set3'] + row['winner_set4'] + row['winner_set5'] -
            row['loser_set1'] - row['loser_set2'] - row['loser_set3'] - row['loser_set4'] - row['loser_set5']
        )
        
        if player1_won:
            Tennis_Player.update_elo(winner = player1, loser = player2, tourney_level = tourney_level, point_diff=point_diff, multiplier_type='log', denom=2.9)
            Tennis_Player.update_surface_elo(winner = player1, loser = player2, surface = surface, tourney_level = tourney_level, point_diff=point_diff, multiplier_type='log', denom=2.9)
        else:
            Tennis_Player.update_elo(winner = player2, loser = player1, tourney_level = tourney_level, point_diff=point_diff, multiplier_type='log', denom=2.9)
            Tennis_Player.update_surface_elo(winner = player2, loser = player1, surface = surface, tourney_level = tourney_level, point_diff=point_diff, multiplier_type='log', denom=2.9)
            

        # head to head
        if player1_won:
            player1.record_h2h_win(player2.name)
        else:
            player2.record_h2h_win(player1.name)

        # last match date
        player1.last_match_date = tourney_date
        player2.last_match_date = tourney_date


    # ========================= Applying Elo Decay at the end also

    last_tourney_date = df['tourney_date'].max()
    for player in players.values():
        Tennis_Player.apply_elo_decay(player, current_date=last_tourney_date)


    # ========================= Encoding Categorical Features if needed with LabelEncoder

    dataset = pd.DataFrame(dataset, columns = features)

    if encode_categories:
        label_encoders = {}

        for col in ['hand_matchup', 'tourney_level', 'surface', 'best_of']:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype(str))
            label_encoders[col] = le

        # Ordinal encoding for tourney round
        round_order = {
            'R128': 1,
            'R64': 2,
            'R32': 3,
            'R16': 4,
            'QF': 5,
            'SF': 6,
            'F': 7
        }

        dataset['tourney_round'] = dataset['tourney_round'].map(round_order)


    return dataset, players
    
def game_differential(df):
    df['winner_games'] = df['winner_set1'] + df['winner_set2'] + df['winner_set3'] + df['winner_set4'] + \
                         df['winner_set5']

    # Calculate number of games won in entire match for loser
    df['loser_games'] = df['loser_set1'] + df['loser_set2'] + df['loser_set3'] + df['loser_set4'] + \
                        df['loser_set5']

    df['total_games'] = df['winner_games'] + df['loser_games']

    df['game_differential'] = abs(df['winner_games'] - df['loser_games'])
    
    return df
# ============= Visualize

#kind of conservative multiplier
def visualize_log_multiplier(df):
    # Average game differential
    game_diff_mean = df['game_differential'].mean()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.array(range(df['game_differential'].min(), df['game_differential'].max() + 1))
    #formulas with different denominators
    # formula = np.log((x/denominator) + 1) bigger win -> bigger multiplier
    a = np.log((x/1) + 1)
    b = np.log((x/2) + 1)
    c = np.log((x/3) + 1)
    d = np.log((x/4) + 1)
    e = np.log((x/5) + 1)

    ax.plot(x, a, 'r', label='1')
    ax.plot(x, b, 'b', label='2')
    ax.plot(x, c, 'g', label='3')
    ax.plot(x, d, 'm', label='4')
    ax.plot(x, e, 'y', label='5')

    fig.suptitle('Natural Logarithm Win Multiplier Denominator', fontsize=14)
    plt.xlabel('Game Differential', fontsize=10)
    plt.ylabel('Multiplier', fontsize=10)

    plt.legend(loc='upper left')

    plt.show()


    print('Win Multiplier When x Equals the Mean Game Differential:')
    print(f'Denominator = 1: {np.log((game_diff_mean/1) + 1)}')
    print(f'Denominator = 2: {np.log((game_diff_mean/2) + 1)}')
    print(f'Denominator = 3: {np.log((game_diff_mean/3) + 1)}')
    print(f'Denominator = 4: {np.log((game_diff_mean/4) + 1)}')
    print(f'Denominator = 5: {np.log((game_diff_mean/5) + 1)}')

    #looking for a denominator that makes the multiplier equal to 1 when x equals the mean game differential
    
    # so basically a baseline for the multiplier
    # game diff bigger than baseline -> more points
    # game diff smaller than baseline -> less points

    ln_optimal_denominator = game_diff_mean / (math.exp(1) - 1)
    print(f'Optimal Denominator Value of Natural Logarithm Win Multiplier: {ln_optimal_denominator}')

#more aggressive multiplier
def visualize_nearly_linear_multiplier(df):
    # Average game differential
    game_diff_mean = df['game_differential'].mean()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.array(range(df['game_differential'].min(), df['game_differential'].max() + 1))
    a = ((x ** 0.5) / 1)
    b = ((x ** 0.5) / 2)
    c = ((x ** 0.5) / 3)
    d = ((x ** 0.5) / 4)
    e = ((x ** 0.5) / 5)

    ax.plot(x, a, 'r', label='1')
    ax.plot(x, b, 'b', label='2')
    ax.plot(x, c, 'g', label='3')
    ax.plot(x, d, 'm', label='4')
    ax.plot(x, e, 'y', label='5')

    fig.suptitle('Nearly Linear Win Multiplier Denominator', fontsize=14)
    plt.xlabel('Game Differential', fontsize=10)
    plt.ylabel('Multiplier', fontsize=10)

    plt.legend(loc='upper left')

    plt.show()

    print('Win Multiplier When x Equals the Mean Game Differential:')
    print(f'Denominator = 1: {((game_diff_mean ** 0.5) / 1)}')
    print(f'Denominator = 2: {((game_diff_mean ** 0.5) / 2)}')
    print(f'Denominator = 3: {((game_diff_mean ** 0.5) / 3)}')
    print(f'Denominator = 4: {((game_diff_mean ** 0.5) / 4)}')
    print(f'Denominator = 5: {((game_diff_mean ** 0.5) / 5)}')

    #looking for a denominator that makes the multiplier equal to 1 when x equals the mean game differential
    
    # so basically a baseline for the multiplier
    # game diff bigger than baseline -> more points
    # game diff smaller than baseline -> less points
    linear_optimal_denominator = math.sqrt(game_diff_mean)

    print(f'Optimal Denominator Value of Linear Win Multiplier: {linear_optimal_denominator}')


# =================

df = game_differential(df)
visualize_log_multiplier(df)
visualize_nearly_linear_multiplier(df)

dataset, players = compute_difference_features(df, start_date = start_date, encode_categories = encode_categories)
    
#dataset.to_csv(f"datasets/ready_atp_matches_{start_date}_2024_difference_{'encoded' if encode_categories else 'not_encoded'}.csv", index = False)