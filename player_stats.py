'''
Simplified version of player class, without elo - related methods
'''

class Player:

	def __init__(self, name):
		self.name = name
		
		# Elo
		self.elo = 1500
		self.surface_elo = {
			'Hard': 1500,
			'Clay': 1500,
			'Grass': 1500
		}

		# Rank
		self.rank = None
		self.rank_points = None

		# Matches
		self.matches = [] # informative 

		# Recent Matches
		self.recent_10 = []
		self.recent_25 = []
		self.recent_50 = []

		self.surface_recent = {
			'Hard': {'10': [], '25': [], '50': []},
			'Clay': {'10': [], '25': [], '50': []},
			'Grass': {'10': [], '25': [], '50': []}
		}

		# Head to Head
		self.h2h_overall = {}
		self.h2h_overall_surface = {}

		self.height = None
		self.hand = None
		self.age = None
		

	@classmethod
	def from_dict(cls, data: dict):
		obj = cls(
			name=data.get('name')
		)

		obj.elo = data.get('elo', 1500)
		obj.surface_elo = data.get('surface_elo', {'Hard': 1500, 'Clay': 1500, 'Grass': 1500})


		return obj
	
	def to_dict(self):
		return {
			'name': self.name,
			'elo': self.elo,
			'surface_elo': self.surface_elo,
			'rank': self.rank,
			'rank_points': self.rank_points,

			'recent_10': self.recent_10,
			'recent_25': self.recent_25,
			'recent_50': self.recent_50,

			'surface_recent': self.surface_recent,

			'height': self.height,
			'hand': self.hand,
			'age': self.age

		}
	
	