import pandas as pd
import json
import numpy as np
import sys
import os
import random
import math
from datetime import datetime
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from player_stats import Player

# =========================

# Elo connected functions
def get_win_multiplier(multiplier_type : str, point_diff:int, winner_elo:float, loser_elo:float, denominator:float):
		'''
			Win multiplier

			Parameters:
				multiplier_type (str): Type of multiplier
				point_diff (int): Point difference
				winner_elo (float): Elo rating of winner
				loser_elo (float): Elo rating of loser
				denominator (float): Denominator

			Returns:
				multiplier (float): Win multiplier
				
			Info:
				2.2 was used by FiveThirtyEight in sports betting as a stabilizing constant
		'''	

		weight = (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))
		
		if multiplier_type == 'log':
			win_multiplier = np.log((point_diff / denominator) + 1) * weight
		elif multiplier_type == 'linear':
			win_multiplier = ((point_diff ** 0.5) / denominator) * weight


		return win_multiplier

def probability_of_winning(player_1_elo : float, player_2_elo : float):
	'''
		Probability of player 1 winning

		Parameters:
			player_1_elo (float): Elo rating of player 1
			player_2_elo (float): Elo rating of player 2

		Returns:
			probability (float): Probability of player 1 winning
	'''
	probability = 1 / (1 + 10 ** ((player_2_elo - player_1_elo) / 400))

	return probability

def set_differential(row:pd.Series):
	winner_sets = row['winner_set1'] + row['winner_set2'] + row['winner_set3'] + row['winner_set4'] + row['winner_set5']
	loser_sets = row['loser_set1'] + row['loser_set2'] + row['loser_set3'] + row['loser_set4'] + row['loser_set5']

	return abs(winner_sets - loser_sets)

def get_optimal_denominator(df: pd.DataFrame, multiplier_type: str):
	game_diff_mean = df['game_differential'].mean()

	if multiplier_type == 'log':
		optimal_denominator = game_diff_mean / (math.exp(1) - 1)
	elif multiplier_type == 'linear':
		optimal_denominator = math.sqrt(game_diff_mean)
	else:
		optimal_denominator = 1
	
	return optimal_denominator

def get_k_factor(player: Player, surface: str, tourney_level: str, tourney_round: str):
	'''

		Parameters:
			player (Player): Player object
			surface (str): Surface
			tourney_level (str): Tourney level
			tourney_round (str): Tourney round

		Returns:
			k_factor (float): K-factor
	
		Info: 
			Im using the formula from Kovalchik (2016), empirical model for K-factor.
			Source:
				Kovalchik, A. (2016). “The K-factor in the Elo rating system.” 
				Journal of Quantitative Analysis in Sports, 12(2), 111–120.

			Formula:
				Ki(t) = 250/(Ni(t) + 5)**0.4
			
			where Ni(t) is the number of matches of player i at time t.

			Plus custom weights for tourney level and tourney round, could be improved or tweaked further
	
	'''
	TOURNEY_LEVEL_WEIGHTS = {
		'G': 1.5,
		'M': 1.2,
		'A': 1.0 
	}

	TOURNEY_ROUND_WEIGHTS = {
		'R128': 0.8,
		'R64': 0.9,
		'R32': 1.0,
		'R16': 1.05,
		'QF': 1.1,
		'SF': 1.2,
		'F': 1.3
	}

	n_matches = len(player.matches) or 1

	surface_matches = [match for match in player.matches if match['surface'] == surface]
	n_surface_matches = len(surface_matches) or 1

	level_weight = TOURNEY_LEVEL_WEIGHTS.get(tourney_level, 1.0)
	round_weight = TOURNEY_ROUND_WEIGHTS.get(tourney_round, 1.0)


	k_factor_base = 250 / (n_matches + 5) ** 0.4
	k_surface_factor_base = 250 / (n_surface_matches + 5) ** 0.4

	#exposure_ratio = n_surface_matches / n_matches
	#surface_k = k_factor * (0.5 + 0.5 * exposure_ratio)

	#really hard to determine so i just use base
	#surface_k = 32

	k_factor = k_factor_base * level_weight * round_weight
	k_surface_factor = k_surface_factor_base * (level_weight * round_weight) # ** 0.5 

	return k_factor, k_surface_factor
	
def update_elo(player: Player, k: float, k_surface: float, win_multiplier: float, score: int, expected_score: float, expected_surface_score: float, surface: str):
	'''
		Update Elo rating

		Parameters:
			player_elo (float): Player's current Elo rating
			k (float): K-factor
			score (int): Actual score (1 for win, 0 for loss)
			expected_score (float): Expected score

		Returns:
			new_elo (float): New Elo rating


		Info:
			Formula I used is the basic elo formula, plus win multiplier for taking into account the point differential,
			k factor based on Kovalchik formula, plus k factor includes tourney level and tourney round weights,
			also surface is entirely on its own
					
	'''

	player_elo = player.elo
	player_surface_elo = player.surface_elo[surface]
		
	new_elo = player_elo + k * (score - expected_score) * win_multiplier
	new_surface_elo = player_surface_elo + k_surface * (score - expected_surface_score) * win_multiplier

	return new_elo, new_surface_elo

	

# Initialization functions
def load_players(year:str):
	players_dict = {}
	
	with open(f"players_elos_from_{year}.json", "r") as f:
		players_dict = json.load(f)

	players = {name: Player.from_dict(data) for name, data in players_dict.items()}
	return players

def game_differential(df):
	df['winner_games'] = df['winner_set1'] + df['winner_set2'] + df['winner_set3'] + df['winner_set4'] + \
						 df['winner_set5']

	# Calculate number of games won in entire match for loser
	df['loser_games'] = df['loser_set1'] + df['loser_set2'] + df['loser_set3'] + df['loser_set4'] + \
						df['loser_set5']

	df['total_games'] = df['winner_games'] + df['loser_games']

	df['game_differential'] = abs(df['winner_games'] - df['loser_games'])
	
	return df


# Player connected functions
def update_after_match(player: Player, surface: str,
	date: int, opponent_name: str,	result: int, tourney_level: str, tourney_round: str		   
	):
	
	# General
	player.matches.append({
		'date': date,
		'opponent': opponent_name,
		'result': result,
		'surface': surface,
		'tourney_level': tourney_level,
		'tourney_round': tourney_round,	
		#'score': score,
	})
	
	if opponent_name not in player.h2h_overall:
		player.h2h_overall[opponent_name] = []
		player.h2h_overall_surface[opponent_name] = {'Hard': [], 'Clay': [], 'Grass': []}
	
	player.h2h_overall[opponent_name].append(result)
	player.h2h_overall_surface[opponent_name][surface].append(result)
	
	# Recent matches
	player.recent_10.append(result)
	player.recent_25.append(result)
	player.recent_50.append(result)
	player.recent_10 = player.recent_10[-10:]
	player.recent_25 = player.recent_25[-25:]
	player.recent_50 = player.recent_50[-50:]

	player.surface_recent[surface]['10'].append(result)
	player.surface_recent[surface]['25'].append(result)
	player.surface_recent[surface]['50'].append(result)
	player.surface_recent[surface]['10'] = player.surface_recent[surface]['10'][-10:]
	player.surface_recent[surface]['25'] = player.surface_recent[surface]['25'][-25:]
	player.surface_recent[surface]['50'] = player.surface_recent[surface]['50'][-50:]

def recent_win_percent_bayesian(player: Player, window: str, fallback: float = 0.5):
	"""
	Calculates the Bayesian-smoothed recent win percentage over a window

	Parameters:
		player (Player): The player
		window (str): '10', '25', or '50'
		fallback (float): Fallback win rate if no matches

	Returns:
		smoothed_win_percent (float): Smoothed win percentage
	"""
	matches = getattr(player, f"recent_{window}", [])

	n = len(matches)
	if n == 0:
		return fallback

	actual = sum(matches) / n
	weight = int(50 / int(window))

	smoothed_win_percent = (weight * fallback + n * actual) / (weight + n)


	return smoothed_win_percent
		
def recent_surface_win_percent_bayesian(player: Player, surface: str, window: str, fallback: float = 0.5):
	"""
	Calculates Bayesian-smoothed surface-specific win percentage

	Parameters:
		player (Player): The player
		surface (str): One of 'Hard', 'Clay', 'Grass'
		window (str): '10', '25', or '50'
		fallback (float): Fallback win rate if no matches

	Returns:
		smoothed_win_percent (float): Smoothed surface win percentage
	"""
	matches = player.surface_recent.get(surface, {}).get(window, [])

	n = len(matches)
	if n == 0:
		return fallback

	actual = sum(matches) / n
	weight = int(50 / int(window))

	smoothed_win_percent = (weight * fallback + n * actual) / (weight + n)
	return smoothed_win_percent

def recent_win_percent_exponential_decay(player: Player, current_date, window=25, decay_lambda=0.01, fallback=0.5):
	matches = player.matches[-window:]
	return decayed_winrate(matches, current_date, decay_lambda=decay_lambda)

def recent_surface_win_percent_exponential_decay(player: Player, surface: str, current_date, window=25, decay_lambda=0.01, fallback=0.5):
	matches = [m for m in player.matches if m['surface'] == surface][-window:]
	return decayed_winrate(matches, current_date, decay_lambda=decay_lambda)

def get_latest_match_date(player: Player):
	if not player.matches:
		return None

	latest_date = max(match['date'] for match in player.matches)
	return latest_date

def get_latest_match_date_on_surface(player: Player, surface: str):
	surface_matches = [match['date'] for match in player.matches if match['surface'] == surface]
	
	if not surface_matches:
		return None

	return max(surface_matches)

def decay_elo(player, current_date, base_elo=1500, long_inactive_base=1000, decay_rate=0.95, threshold_days=360, threshold_surface_days=720, long_inactive_days=5*365):
	'''
		Decay Elo rating

		Parameters:
			player (Player): the player
			current_date (int): Current date, format YYYYMMDD
			base_elo (float): Base Elo, should be 1500
			long_inactive_base (float): if inactive for longer than long_inactive_days, decay to this
			decay_rate (float): Decay rate
			threshold_days (int): Threshold days for decay
			threshold_surface_days (int): Threshold surface days for decay
			long_inactive_days (int): if inactive for longer than this, decay to long_inactive_base

		Returns:
			None

		Info:
			Decaying elo based on latest match, plus decay based on inactivity, plus decay based on surface inactivity
			Also decay rank points and rank
	'''
	
	if not player.matches:
		return

	current = datetime.strptime(str(current_date), "%Y%m%d")
	
	# overall Elo decay based on latest match
	last_date = max(int(match['date']) for match in player.matches)
	last_date_dt = datetime.strptime(str(last_date), "%Y%m%d")
	delta_days = (current - last_date_dt).days

	if delta_days > threshold_days:
		months_inactive = (delta_days - threshold_days) / 30

		if delta_days > long_inactive_days:
			decay_factor = decay_rate ** ((delta_days - long_inactive_days) / 30)
			player.elo = long_inactive_base + (player.elo - long_inactive_base) * decay_factor
			# player.elo = player.elo * decay_factor
		else:
			decay_factor = decay_rate ** months_inactive
			player.elo = base_elo + (player.elo - base_elo) * decay_factor
			# player.elo = player.elo * decay_factor

		# Rank Points
		if player.rank_points is not None:
			player.rank_points = max(0, player.rank_points * decay_factor)

		# Rank — increase toward 999
		if player.rank is not None:
			rank_gap = 999 - player.rank
			player.rank = min(999, player.rank + rank_gap * (1 - decay_factor))

	# surface elo decay based on latest match on surface
	for surface in player.surface_elo:
		surface_dates = [int(match['date']) for match in player.matches if match['surface'] == surface]
		if not surface_dates:
			continue

		last_surface_date = max(surface_dates)
		last_surface_dt = datetime.strptime(str(last_surface_date), "%Y%m%d")
		delta_surface_days = (current - last_surface_dt).days

		if delta_surface_days > threshold_surface_days:
			months_surface_inactive = (delta_surface_days - threshold_surface_days) / 30

			if delta_surface_days > long_inactive_days:
				decay_factor_surface = decay_rate ** ((delta_surface_days - long_inactive_days) / 30)
				player.surface_elo[surface] = long_inactive_base + (player.surface_elo[surface] - long_inactive_base) * decay_factor_surface
				#player.surface_elo[surface] = player.surface_elo[surface] * decay_factor_surface
			else:
				decay_factor_surface = decay_rate ** months_surface_inactive
				player.surface_elo[surface] = base_elo + (player.surface_elo[surface] - base_elo) * decay_factor_surface
				# player.surface_elo[surface] = player.surface_elo[surface] * decay_factor_surface

def print_top_players(players: dict):
	sorted_players = sorted(players.values(), key=lambda p: p.elo, reverse=True)
	print("Top 20 players by Elo:")
	for player in sorted_players[:20]:
		print(f"{player.name}: Elo={player.elo:.2f}, Surface Elo={player.surface_elo}")

def get_h2h_win_percent(player1: Player, player2: Player, weight: int = 2, fallback: float = 0.5):
	results = player1.h2h_overall.get(player2.name, [])
	if not results:
		return fallback

	wins = sum(results)	
	total = len(results)
	smoothed = (wins + weight * fallback) / (total + weight)
	return smoothed

def get_surface_h2h_win_percent(player1: Player, player2: Player, surface: str, weight: int = 2, fallback: float = 0.5):
	p1_matches = player1.h2h_overall_surface.get(player2.name, {}).get(surface, [])

	if not p1_matches:
		return fallback

	p1_wins = sum(p1_matches)
	total = len(p1_matches)

	smoothed_pct = (p1_wins + weight * fallback) / (total + weight)
	return smoothed_pct

def decayed_winrate(matches, current_date, decay_lambda=0.01):
	"""
	Calculate exponentially decayed winrate

	Parameters:
		matches (list of dicts): player matches
		current_date (int): Current date as YYYYMMDD
		decay_lambda (float): decay lambda, higher means faster decay just set it to 0.01

	Returns:
		decayed_winrate (float): Decayed winrate
	"""
	if not matches:
		return 0.5

	current_dt = datetime.strptime(str(current_date), "%Y%m%d")

	weighted_sum = 0.0
	weight_total = 0.0
	last_date = 0

	for match in matches:
		result = match['result']
		match_dt = datetime.strptime(str(match['date']), "%Y%m%d")
		days_ago = (current_dt - match_dt).days
		last_date = max(last_date, (current_dt - match_dt).days)

		weight = math.exp(-decay_lambda * days_ago)
		weighted_sum += result * weight
		weight_total += weight

	# fallback only if weights are too tiny, otherwise it will be 0.5
	if weight_total < 0.001:
		return 0.0

	return weighted_sum / weight_total





# Main functions
def compute_features(df:pd.DataFrame, players:dict):

	df = game_differential(df)

	dataset = []
	
	features = [
		'player1_welo', 'player2_welo',
		'player1_surface_welo', 'player2_surface_welo',

		'player1_rank', 'player2_rank',
		'player1_rank_points', 'player2_rank_points',

		'player1_recent_10_exponential_decay', 'player1_recent_25_exponential_decay', 'player1_recent_50_exponential_decay',
		'player2_recent_10_exponential_decay', 'player2_recent_25_exponential_decay', 'player2_recent_50_exponential_decay',
		'player1_recent_10_surface_exponential_decay', 'player1_recent_25_surface_exponential_decay', 'player1_recent_50_surface_exponential_decay',
		'player2_recent_10_surface_exponential_decay', 'player2_recent_25_surface_exponential_decay', 'player2_recent_50_surface_exponential_decay',

		'player1_age', 'player2_age',
		'player1_height', 'player2_height',
		'player1_hand', 'player2_hand',

		'surface', 'tourney_level', 'tourney_round', 'best_of',

		'player1_won'
	]

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

		# ========== Decay Elo

		decay_elo(player1, tourney_date)
		decay_elo(player2, tourney_date)
		
		# ========== Features

		player1_welo = player1.elo
		player2_welo = player2.elo
		player1_surface_welo = player1.surface_elo[surface]
		player2_surface_welo = player2.surface_elo[surface]
		player1_rank = row['winner_rank'] if row['winner_name'] == player1.name else row['loser_rank']
		player2_rank = row['winner_rank'] if row['winner_name'] == player2.name else row['loser_rank']
		player1_rank_points = row['winner_rank_points'] if row['winner_name'] == player1.name else row['loser_rank_points']
		player2_rank_points = row['winner_rank_points'] if row['winner_name'] == player2.name else row['loser_rank_points']
		
		
		player1_height = row['winner_ht'] if row['winner_name'] == player1.name else row['loser_ht']
		player2_height = row['winner_ht'] if row['winner_name'] == player2.name else row['loser_ht']
		player1_age = row['winner_age'] if row['winner_name'] == player1.name else row['loser_age']
		player2_age = row['winner_age'] if row['winner_name'] == player2.name else row['loser_age']
		player1_hand = row['winner_hand'] if row['winner_name'] == player1.name else row['loser_hand']
		player2_hand = row['winner_hand'] if row['winner_name'] == player2.name else row['loser_hand']
		
		tourney_level = row['tourney_level']
		tourney_round = row['round']
		best_of = row['best_of']

		player1_h2h_overall_pct = get_h2h_win_percent(player1, player2)
		player1_h2h_overall_surface_pct = get_surface_h2h_win_percent(player1, player2, surface)
		player1_won = 1 if row['winner_name'] == player1.name else 0

		'''
		player1_recent_10 = recent_win_percent_bayesian(player1, '10')
		player1_recent_25 = recent_win_percent_bayesian(player1, '25')
		player1_recent_50 = recent_win_percent_bayesian(player1, '50')
		player2_recent_10 = recent_win_percent_bayesian(player2, '10')
		player2_recent_25 = recent_win_percent_bayesian(player2, '25')
		player2_recent_50 = recent_win_percent_bayesian(player2, '50')
		player1_recent_10_surface = recent_surface_win_percent_bayesian(player1, surface, '10')
		player1_recent_25_surface = recent_surface_win_percent_bayesian(player1, surface, '25')
		player1_recent_50_surface = recent_surface_win_percent_bayesian(player1, surface, '50')
		player2_recent_10_surface = recent_surface_win_percent_bayesian(player2, surface, '10')
		player2_recent_25_surface = recent_surface_win_percent_bayesian(player2, surface, '25')
		player2_recent_50_surface = recent_surface_win_percent_bayesian(player2, surface, '50')
		'''

		player1_recent_10_exponential_decay = recent_win_percent_exponential_decay(player1, tourney_date, window=25, decay_lambda=0.01)
		player1_recent_25_exponential_decay = recent_win_percent_exponential_decay(player1, tourney_date, window=25, decay_lambda=0.01)
		player1_recent_50_exponential_decay = recent_win_percent_exponential_decay(player1, tourney_date, window=25, decay_lambda=0.01)
		player2_recent_10_exponential_decay = recent_win_percent_exponential_decay(player2, tourney_date, window=25, decay_lambda=0.01)
		player2_recent_25_exponential_decay = recent_win_percent_exponential_decay(player2, tourney_date, window=25, decay_lambda=0.01)
		player2_recent_50_exponential_decay = recent_win_percent_exponential_decay(player2, tourney_date, window=25, decay_lambda=0.01)

		player1_recent_10_surface_exponential_decay = recent_surface_win_percent_exponential_decay(player1, surface, tourney_date, window=25, decay_lambda=0.01)
		player1_recent_25_surface_exponential_decay = recent_surface_win_percent_exponential_decay(player1, surface, tourney_date, window=25, decay_lambda=0.01)
		player1_recent_50_surface_exponential_decay = recent_surface_win_percent_exponential_decay(player1, surface, tourney_date, window=25, decay_lambda=0.01)
		player2_recent_10_surface_exponential_decay = recent_surface_win_percent_exponential_decay(player2, surface, tourney_date, window=25, decay_lambda=0.01)
		player2_recent_25_surface_exponential_decay = recent_surface_win_percent_exponential_decay(player2, surface, tourney_date, window=25, decay_lambda=0.01)
		player2_recent_50_surface_exponential_decay = recent_surface_win_percent_exponential_decay(player2, surface, tourney_date, window=25, decay_lambda=0.01)

		
		new_dataset_row = [
			player1_welo, player2_welo,
			player1_surface_welo, player2_surface_welo,

			player1_rank, player2_rank,
			player1_rank_points, player2_rank_points,

			player1_recent_10_exponential_decay, player1_recent_25_exponential_decay, player1_recent_50_exponential_decay,
			player2_recent_10_exponential_decay, player2_recent_25_exponential_decay, player2_recent_50_exponential_decay,
			player1_recent_10_surface_exponential_decay, player1_recent_25_surface_exponential_decay, player1_recent_50_surface_exponential_decay,
			player2_recent_10_surface_exponential_decay, player2_recent_25_surface_exponential_decay, player2_recent_50_surface_exponential_decay,
			
			player1_age, player2_age,
			player1_height, player2_height,
			player1_hand, player2_hand,

			surface, tourney_level, tourney_round, best_of,

			player1_won
		]

		dataset.append(new_dataset_row)

		# =========== Update Elo

		winner = player1 if row['winner_name'] == player1.name else player2
		loser = player1 if row['winner_name'] == player2.name else player2
		
		set_diff = set_differential(row)
		expected_win_prob = probability_of_winning(winner.elo, loser.elo)
		expected_lose_prob = 1 - expected_win_prob
		expected_surface_prob = probability_of_winning(winner.surface_elo[surface], loser.surface_elo[surface])
		expected_surface_lose_prob = 1 - expected_surface_prob


		optimal_denominator = get_optimal_denominator(df, multiplier_type = 'log')
		win_multiplier = get_win_multiplier(multiplier_type = 'log', point_diff = set_diff, winner_elo = winner.elo, loser_elo = loser.elo, denominator = optimal_denominator)


		'''
			Using zero-sum system, i would need to harmonize k-factors so delta elo changes are the same
			
			But on the other hand, using non-zero-sum system can help to model better improving rate,
			and generally is better as a feature I think

			
		'''
		k_factor_winner, k_factor_surface_winner = get_k_factor(winner, surface, row['tourney_level'], row['round'])
		k_factor_loser, k_factor_surface_loser = get_k_factor(loser, surface, row['tourney_level'], row['round'])


		winner_new_elo, winner_new_surface_elo = update_elo(winner, k_factor_winner, k_factor_surface_winner, win_multiplier, 1, expected_win_prob, expected_surface_prob, surface)
		loser_new_elo, loser_new_surface_elo = update_elo(loser, k_factor_loser, k_factor_surface_loser, win_multiplier, 0, expected_lose_prob, expected_surface_lose_prob, surface)


		# ========== Update Player
		winner.elo = winner_new_elo
		loser.elo = loser_new_elo
		winner.surface_elo[surface] = winner_new_surface_elo
		loser.surface_elo[surface] = loser_new_surface_elo

		update_after_match(winner, surface, tourney_date, loser_name, 1, row['tourney_level'], row['round'])
		update_after_match(loser, surface, tourney_date, winner_name, 0, row['tourney_level'], row['round'])

		winner.rank = row['winner_rank']
		loser.rank = row['loser_rank']
		winner.rank_points = row['winner_rank_points']
		loser.rank_points = row['loser_rank_points']

		winner.height = row['winner_ht']
		loser.height = row['loser_ht']
		winner.age = row['winner_age']
		loser.age = row['loser_age']
		winner.hand = row['winner_hand']
		loser.hand = row['loser_hand']

		

	# ========== Final Decay
	final_date = df['tourney_date'].max()
	for player in players.values():
		decay_elo(player, final_date)

	dataset = pd.DataFrame(dataset, columns = features)
	return dataset, players

# =========================

if __name__ == "__main__":
	df = pd.read_csv("datasets/cleaned_atp_matches_2010_2024.csv")
	players = load_players(year = '2010')

	dataset, players = compute_features(df, players)
	dataset.to_csv("datasets/elo_driven_features_2010_2024.csv", index = False)
	
	#joblib.dump(dataset.columns.tolist(), "models/elo_driven_columns.pkl")
	#print_top_players(players)

	players_json = {name: player.to_dict() for name, player in players.items()}
	output_path = f"players_output_2010_2024.json"
	with open(output_path, "w") as f:
		json.dump(players_json, f, indent=4)





