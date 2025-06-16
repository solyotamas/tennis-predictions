import pandas as pd
import random
import re

df = pd.read_csv("datasets/all_atp_2000_2024.csv")

# =========================

def is_valid_score(score):
    return bool(re.search(r'\d+-\d+', str(score)))

def parse_score(score):
    score = re.sub(r'\([^\)]*\)', '', str(score))
    score = re.sub(r'\[[^\]]*\]', '', score)
    sets = score.strip().split()
    winner_sets = []
    loser_sets = []
    for s in sets:
        if '-' in s:
            w, l = s.split('-')[:2]
            w = re.sub(r'\D', '', w)
            l = re.sub(r'\D', '', l)
            winner_sets.append(int(w) if w else 0)
            loser_sets.append(int(l) if l else 0)
    while len(winner_sets) < 5:
        winner_sets.append(0)
        loser_sets.append(0)
    return winner_sets + loser_sets

def clean_data(df: pd.DataFrame):
	
	columns = [
		'tourney_date', 'tourney_level', 'surface',
		'winner_name', 'winner_age','winner_ht', 'winner_hand' ,'winner_rank', 'winner_rank_points', 
		'loser_name', 'loser_age', 'loser_ht', 'loser_hand', 'loser_rank', 'loser_rank_points',
		'score', 'best_of', 'round'
	]

	df = df[columns]

	# =========================

	# Tourney Level
	tourney_level_keep = ['A', 'M', 'G']
	df = df[df['tourney_level'].isin(tourney_level_keep)]

	# Surface
	surface_keep = ['Hard', 'Clay', 'Grass']
	df = df[df['surface'].isin(surface_keep)]

	# Hand
	hand_keep = ['R', 'L', 'U']
	df = df[df['winner_hand'].isin(hand_keep)]
	df = df[df['loser_hand'].isin(hand_keep)]

	# Height
	all_heights = pd.concat([df['winner_ht'], df['loser_ht']])
	median_height = all_heights.median()
	df['winner_ht'] = df['winner_ht'].fillna(median_height)
	df['loser_ht'] = df['loser_ht'].fillna(median_height)

	# Rank
	df = df.dropna(subset=['winner_rank', 'loser_rank'])

	# Score
	df = df[df['score'].apply(is_valid_score)]
	set_cols = [f'winner_set{i+1}' for i in range(5)] + [f'loser_set{i+1}' for i in range(5)]
	df[set_cols] = df['score'].apply(parse_score).apply(pd.Series)
	df = df.drop(columns=['score'])
      

	# Round
	round_drop = 'BR'
	df = df[df['round'] != round_drop]

	# =========================
	print(df.isna().sum())
	
	return df

# =========================

df = clean_data(df)
df.to_csv("datasets/cleaned_atp_matches_2000_2024.csv", index=False)