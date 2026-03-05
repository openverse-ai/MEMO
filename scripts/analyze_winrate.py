import json
import matplotlib.pyplot as plt
import sys

# Get file path from command line or use default
if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    json_file = "/teamspace/studios/this_studio/new/unstable_baseline_neg/mpr/logs/20250923_184113_pureguide_noreplay_briscola_sample_gpt_noadd/trajectories/gen4_trajectories_gen4_vs_best.json"

# Read the JSON file
with open(json_file, 'r') as f:
    games = json.load(f)

print(f"Total games: {len(games)}")

# Track wins and calculate win rate
wins = 0
losses = 0
win_rates = []
game_numbers = []

for i, game in enumerate(games):
    # Find the index of the "best_" agent
    best_agent_idx = None
    for idx, agent_name in enumerate(game["agent_names"]):
        if agent_name.startswith("best_"):
            best_agent_idx = idx
            break
    
    if best_agent_idx is None:
        print(f"Warning: No 'best_' agent found in game {i+1}")
        continue
    
    # Check if the best agent won
    if best_agent_idx in game["winners"]:
        wins += 1
    else:
        losses += 1
    
    # Calculate win rate
    total_games = wins + losses
    win_rate = (wins / total_games) * 100
    
    win_rates.append(win_rate)
    game_numbers.append(i + 1)
    
    # Print progress every 100 games
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1} games: {wins} wins, {losses} losses, win rate: {win_rate:.2f}%")

# Final statistics
print(f"\nFinal statistics:")
print(f"Total games analyzed: {len(game_numbers)}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Final win rate: {win_rates[-1]:.2f}%")

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(game_numbers, win_rates, 'b-', linewidth=1)
plt.xlabel('Game Number', fontsize=12)
plt.ylabel('Win Rate (%)', fontsize=12)
plt.title('Win Rate of Best Agent vs Game Number', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(1, len(game_numbers))
plt.ylim(0, 100)

# Add horizontal line at 50%
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% win rate')
plt.legend()

# Save the plot
plt.savefig('winrate_plot.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'winrate_plot.png'")

# Also display the plot
plt.show()
