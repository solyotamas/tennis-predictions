import json

# load players
with open("players_output_2010_2024.json", "r") as f:
    players_dict = json.load(f)

# sort
sorted_players = sorted(players_dict.values(), key=lambda p: p['elo'], reverse=True)

# print
print("Top 10 players by Elo:")
for player in sorted_players[:20]:
    print(f"{player['name']}: Elo={player['elo']:.2f}")
