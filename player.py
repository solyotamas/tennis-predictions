from datetime import datetime
import numpy as np

class Tennis_Player:

	def __init__(self, name):
		self.name = name
		
		# Elo
		self.elo = 1500
		self.surface_elo = {
			'Hard': 1500,
			'Clay': 1500,
			'Grass': 1500
		}

		self.last_match_date = None


		# Matches
		self.recent_10 = []
		self.recent_25 = []
		self.recent_50 = []

		self.surface_recent = {
			'Hard': {'10': [], '25': [], '50': []},
			'Clay': {'10': [], '25': [], '50': []},
			'Grass': {'10': [], '25': [], '50': []}
		}

		# Head to Head
		self.h2h_wins = {}


		# Just for testing
		self.hand = None
		self.age = None
		self.height = None
		self.rank = None
		self.rank_points = None


	# with Bayesian smoothing bc just falling back to 0.5 is pretty inaccurate 
	def win_pct(self, matches: list, weight: int, fallback: int = 0.5):
		n = len(matches)
		if n == 0:
			return fallback

		actual = sum(matches) / n
		return (weight * fallback + n * actual) / (weight + n)
		

	# with Bayesian smoothing also
	def surface_win_pct(self, surface: str, window: str, weight: int, fallback: int = 0.5):
		results = self.surface_recent.get(surface, {}).get(window, [])

		n = len(results)
		if n == 0:
			return fallback

		actual = sum(results) / n
		return (weight * fallback + n * actual) / (weight + n)

		 


	def update_after_match(self, won: bool, surface: str):
		outcome = 1 if won else 0

		# General
		self.recent_10.append(outcome)
		self.recent_25.append(outcome)
		self.recent_50.append(outcome)

		self.recent_10 = self.recent_10[-10:]
		self.recent_25 = self.recent_25[-25:]
		self.recent_50 = self.recent_50[-50:]

		# Surface
		self.surface_recent[surface]['10'].append(outcome)
		self.surface_recent[surface]['25'].append(outcome)
		self.surface_recent[surface]['50'].append(outcome)

		self.surface_recent[surface]['10'] = self.surface_recent[surface]['10'][-10:]
		self.surface_recent[surface]['25'] = self.surface_recent[surface]['25'][-25:]
		self.surface_recent[surface]['50'] = self.surface_recent[surface]['50'][-50:]
		

		
	def record_h2h_win(self, opponent_name: str):
		self.h2h_wins[opponent_name] = self.h2h_wins.get(opponent_name, 0) + 1
		

	def get_h2h_diff(self, opponent):
		my_wins = self.h2h_wins.get(opponent.name, 0)
		their_wins = opponent.h2h_wins.get(self.name, 0)
		return my_wins - their_wins


	def to_dict(self):
		return {
			'name': self.name,
			'elo': self.elo,
			'surface_elo': self.surface_elo,
			
		}
	
	def to_final_dict(self):
		return {
			'name': self.name,
			'elo': self.elo,
			'surface_elo': self.surface_elo,
			'recent_10': self.recent_10,
			'recent_25': self.recent_25,
			'recent_50': self.recent_50,
			'surface_recent': self.surface_recent,
			'h2h_wins': self.h2h_wins,
			'hand': self.hand,
			'age': self.age,
			'height': self.height,
			'rank': self.rank,
			'rank_points': self.rank_points,
		}
	
	@classmethod
	def from_dict(cls, data: dict):
		obj = cls(
			name=data.get('name')
		)

		obj.elo = data.get('elo', 1500)
		obj.surface_elo = data.get('surface_elo', {'Hard': 1500, 'Clay': 1500, 'Grass': 1500})


		return obj
	

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
		'''

		#2.2 was used by FiveThirtyEight in sports betting as a stabilizing constant

		weight = (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))
		
		if multiplier_type == 'log':
			win_multiplier = np.log((point_diff / denominator) + 1) * weight
		elif multiplier_type == 'linear':
			win_multiplier = ((point_diff ** 0.5) / denominator) * weight


		return win_multiplier

	
	@staticmethod
	def apply_elo_decay(player: 'Tennis_Player', current_date, base_elo = 1500 ,decay_rate=5.0, threshold_days=180, threshold_surface_days=720):
		if not player.matches:
			return

		current = datetime.strptime(str(current_date), "%Y%m%d")
		
		# Overall Elo decay based on latest match
		last_date = max(int(match['date']) for match in player.matches)
		last_date_dt = datetime.strptime(str(last_date), "%Y%m%d")
		delta_days = (current - last_date_dt).days

		if delta_days > threshold_days:
			months_inactive = (delta_days - threshold_days) / 30
			decay_factor = decay_rate ** months_inactive
			player.elo = base_elo + (player.elo - base_elo) * decay_factor

			# Rank Points
			if player.rank_points is not None:
				player.rank_points = max(0, player.rank_points * decay_factor)

			# Rank — increase toward 999
			if player.rank is not None:
				rank_gap = 999 - player.rank
				player.rank = min(999, player.rank + rank_gap * (1 - decay_factor))

			player.recent_10.clear()
			player.recent_25.clear()
			player.recent_50.clear()
			for surface_data in player.surface_recent.values():
				surface_data['10'].clear()
				surface_data['25'].clear()
				surface_data['50'].clear()


		
		# Decaying surface elo separately
		for surface in player.surface_elo:
			surface_dates = [int(match['date']) for match in player.matches if match['surface'] == surface]
			if not surface_dates:
				continue

			last_surface_date = max(surface_dates)
			last_surface_dt = datetime.strptime(str(last_surface_date), "%Y%m%d")
			delta_surface_days = (current - last_surface_dt).days

			if delta_surface_days > threshold_surface_days:
				months_surface_inactive = (delta_surface_days - threshold_surface_days) / 30
				decay_factor_surface = decay_rate ** months_surface_inactive
				player.surface_elo[surface] = base_elo + (player.surface_elo[surface] - base_elo) * decay_factor_surface

		
	# Updated to using weighted elo, which basically means it also
	# considers how dominant the win was, multiplier in feature computing scripts

	# Also trying to tweak k factor more, could be improved like bigger k for earlier in the carrier
	# or / and with some combination on tourney level and tourney round 
	# current is just really simple 
	@staticmethod
	def update_elo(winner: 'Tennis_Player', loser: 'Tennis_Player', tourney_level: str, 
                point_diff, multiplier_type='log', denom=2.9):
		
		Tourney_codes = {
			'G': 40,
			'M': 32,
			'A': 24
		}

		win_mult = Tennis_Player.get_win_multiplier(multiplier_type, point_diff, winner.elo, loser.elo, denom)
    	
		k = Tourney_codes.get(tourney_level, 24) * win_mult

		expected_win = 1 / (1 + 10 ** ((loser.elo - winner.elo) / 400))
		change = k * (1 - expected_win)

		winner.elo += change
		loser.elo -= change


	# Same as update_elo, but for surface elo
	@staticmethod
	def update_surface_elo(winner: 'Tennis_Player', loser: 'Tennis_Player', surface: str, tourney_level: str, point_diff, multiplier_type='log', denom=2.9):

		Tourney_codes = {
			'G': 40,
			'M': 32,
			'A': 24
		}

		win_mult = Tennis_Player.get_win_multiplier(multiplier_type, point_diff, winner.elo, loser.elo, denom)
		k = Tourney_codes.get(tourney_level, 24) * win_mult

		winner_surface_elo = winner.surface_elo[surface]
		loser_surface_elo = loser.surface_elo[surface]



		expected_win = 1 / (1 + 10 ** ((loser_surface_elo - winner_surface_elo) / 400))
		change = k * (1 - expected_win)

		winner.surface_elo[surface] += change
		loser.surface_elo[surface] -= change
	
	
		


	 