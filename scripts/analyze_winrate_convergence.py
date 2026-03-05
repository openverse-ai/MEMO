import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def calculate_binomial_confidence_interval(wins, total, confidence=0.95):
    """
    Calculate confidence interval for binomial proportion using Wilson score interval.
    This is more accurate than normal approximation, especially for small samples or extreme proportions.
    """
    from scipy import stats
    
    if total == 0:
        return 0, 0, 0
    
    p_hat = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return p_hat, lower, upper

def detect_convergence(win_rates, window_size=20, threshold=0.02):
    """
    Detect when win rate has converged by checking if it stays within a range.
    
    Args:
        win_rates: List of win rates
        window_size: Number of games to check for stability
        threshold: Maximum allowed variation (as fraction, e.g., 0.02 = 2%)
    
    Returns:
        convergence_point: Index where convergence is detected (or None)
        convergence_range: (lower, upper) bounds of convergence
    """
    if len(win_rates) < window_size:
        return None, None
    
    for i in range(window_size, len(win_rates)):
        window = win_rates[i-window_size:i]
        window_mean = np.mean(window)
        window_std = np.std(window)
        
        # Check if all values in window are within threshold of the mean
        max_deviation = max(abs(val - window_mean) for val in window) / 100  # Convert to fraction
        
        if max_deviation <= threshold:
            return i - window_size, (window_mean - threshold*100, window_mean + threshold*100)
    
    return None, None

# Get file path from command line or use default
if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    json_file = "/teamspace/studios/this_studio/new/unstable_baseline_neg/logs/20250924_035720_pureguide_noreplay_connectfour_sample_gpt/trajectories/gen1_trajectories_gen1_vs_best.json"

# Read the JSON file
with open(json_file, 'r') as f:
    games = json.load(f)

print(f"Total games: {len(games)}")

# Track wins and calculate win rate
wins = 0
losses = 0
win_rates = []
game_numbers = []
win_counts = []
confidence_lower = []
confidence_upper = []

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
    
    # Calculate win rate and confidence interval
    total_games = wins + losses
    win_rate, lower, upper = calculate_binomial_confidence_interval(wins, total_games)
    
    win_rates.append(win_rate * 100)
    confidence_lower.append(lower * 100)
    confidence_upper.append(upper * 100)
    game_numbers.append(i + 1)
    win_counts.append(wins)
    
    # Print progress every 100 games
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1} games: {wins} wins, {losses} losses, win rate: {win_rate*100:.2f}%")

# Detect convergence
convergence_point, convergence_range = detect_convergence(win_rates, window_size=30, threshold=0.02)

# Final statistics
print(f"\nFinal statistics:")
print(f"Total games analyzed: {len(game_numbers)}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Final win rate: {win_rates[-1]:.2f}%")
print(f"95% Confidence Interval: [{confidence_lower[-1]:.2f}%, {confidence_upper[-1]:.2f}%]")

if convergence_point is not None:
    print(f"\nConvergence detected at game {game_numbers[convergence_point]}")
    print(f"Convergence range: {convergence_range[0]:.2f}% - {convergence_range[1]:.2f}%")
else:
    print("\nNo convergence detected within the threshold criteria")

# Calculate standard error for binomial proportion
final_se = np.sqrt((win_rates[-1]/100 * (1 - win_rates[-1]/100)) / len(games)) * 100
print(f"\nStandard Error (binomial): ±{final_se:.2f}%")

# Create the plot
plt.figure(figsize=(14, 10))

# Main win rate line
plt.plot(game_numbers, win_rates, 'b-', linewidth=1.5, label='Win Rate', alpha=0.8)

# Confidence interval band
plt.fill_between(game_numbers, confidence_lower, confidence_upper, 
                 alpha=0.2, color='blue', label='95% Confidence Interval')

# Add convergence indicators
if convergence_point is not None:
    # Convergence range
    plt.axhspan(convergence_range[0], convergence_range[1], 
                alpha=0.1, color='green', label=f'Convergence Range (±2%)')
    
    # Vertical line at convergence point
    plt.axvline(x=game_numbers[convergence_point], color='green', 
                linestyle='--', alpha=0.7, 
                label=f'Convergence at Game {game_numbers[convergence_point]}')
    
    # Add text annotation
    plt.annotate(f'Convergence\nGame {game_numbers[convergence_point]}', 
                xy=(game_numbers[convergence_point], win_rates[convergence_point]),
                xytext=(game_numbers[convergence_point] + 10, win_rates[convergence_point] + 5),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, ha='left')

# Add horizontal line at 50%
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% win rate')

# Add final win rate with error bars
final_rate = win_rates[-1]
plt.axhline(y=final_rate, color='orange', linestyle='-', alpha=0.5, 
            label=f'Final Rate: {final_rate:.2f}%')
plt.axhspan(final_rate - final_se, final_rate + final_se, 
            alpha=0.1, color='orange', label=f'±1 SE ({final_se:.2f}%)')

plt.xlabel('Game Number', fontsize=12)
plt.ylabel('Win Rate (%)', fontsize=12)
plt.title('Win Rate Convergence Analysis for Best Agent', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(1, len(game_numbers))
plt.ylim(0, max(100, max(confidence_upper) + 5))
plt.legend(loc='best', fontsize=10)

# Save the plot
plt.savefig('winrate_convergence_plot.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'winrate_convergence_plot.png'")

# Create a second plot showing rolling statistics
plt.figure(figsize=(14, 8))

# Calculate rolling mean and std
window = 20
rolling_mean = []
rolling_std = []

for i in range(len(win_rates)):
    if i < window:
        rolling_mean.append(np.mean(win_rates[:i+1]))
        rolling_std.append(np.std(win_rates[:i+1]))
    else:
        rolling_mean.append(np.mean(win_rates[i-window+1:i+1]))
        rolling_std.append(np.std(win_rates[i-window+1:i+1]))

plt.subplot(2, 1, 1)
plt.plot(game_numbers, win_rates, 'b-', alpha=0.3, linewidth=1, label='Actual Win Rate')
plt.plot(game_numbers, rolling_mean, 'r-', linewidth=2, label=f'{window}-Game Rolling Average')
plt.xlabel('Game Number')
plt.ylabel('Win Rate (%)')
plt.title(f'Win Rate with {window}-Game Rolling Average')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(game_numbers, rolling_std, 'g-', linewidth=2)
plt.xlabel('Game Number')
plt.ylabel('Rolling Standard Deviation (%)')
plt.title(f'{window}-Game Rolling Standard Deviation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('winrate_rolling_stats.png', dpi=300, bbox_inches='tight')
print(f"Rolling statistics plot saved as 'winrate_rolling_stats.png'")

# Print explanation about confidence intervals
print("\n" + "="*60)
print("STATISTICAL EXPLANATION: How to Calculate ± Standard Deviation")
print("="*60)
print("\nFor win rate convergence, we use binomial confidence intervals because:")
print("1. Each game is a binary outcome (win/loss)")
print("2. We're estimating a proportion (win rate)")
print("\nMethods used:")
print("1. **Wilson Score Interval**: More accurate than normal approximation,")
print("   especially for small samples or extreme proportions")
print("2. **Standard Error**: SE = sqrt(p(1-p)/n)")
print(f"   For our data: SE = sqrt({win_rates[-1]/100:.3f} × {1-win_rates[-1]/100:.3f} / {len(games)}) = {final_se:.2f}%")
print("\n3. **Convergence Detection**: Win rate is considered converged when")
print("   it stays within ±2% for at least 30 consecutive games")
print("\n4. **Rolling Window Analysis**: Shows local variability over time")

plt.show()

