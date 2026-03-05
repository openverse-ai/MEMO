"""
Tournament Scheduler - Creates fair round-robin schedules
"""

import itertools
from typing import List, Tuple


def create_round_robin_schedule(
    agents: List[str],
    num_players_per_game: int,
    num_rounds: int = 1
) -> List[Tuple[str, ...]]:
    """
    Create a fair round-robin tournament schedule.
    Each combination of players plays with all position orderings.
    
    Examples:
    - 5 agents, 2 players per game, 1 round:
      Each pair plays twice (positions swapped)
      Total: C(5,2) * 2 = 10 * 2 = 20 games
      
    - 4 agents, 3 players per game, 1 round:
      Each triple plays 6 times (all position permutations)
      Total: C(4,3) * 6 = 4 * 6 = 24 games
      
    - 5 agents, 2 players per game, 2 rounds:
      Each pair plays 4 times (2 position swaps * 2 rounds)
      Total: C(5,2) * 2 * 2 = 10 * 2 * 2 = 40 games
    
    Args:
        agents: List of all agent names in the tournament
        num_players_per_game: Number of players in each game
        num_rounds: Number of complete round-robin rounds
        
    Returns:
        List of matches (agent tuples in play order)
    """
    matches = []
    
    for _ in range(num_rounds):
        # Get all combinations of agents for games of size num_players_per_game
        for combo in itertools.combinations(agents, num_players_per_game):
            # For each combination, create all position permutations for fairness
            for perm in itertools.permutations(combo):
                matches.append(perm)
    
    return matches


def create_limited_schedule(
    agents: List[str],
    num_players_per_game: int,
    num_rounds: int = 1,
    positions_per_combo: int = 2
) -> List[Tuple[str, ...]]:
    """
    Create a limited schedule when full permutations would be too many.
    
    Example:
    - 10 agents, 5 players per game, 1 round, 2 positions per combo:
      Each 5-player combination plays only 2 position variations
      Total: C(10,5) * 2 = 252 * 2 = 504 games
      (instead of 252 * 120 = 30,240 games with all permutations)
    
    Args:
        agents: List of all agent names
        num_players_per_game: Players per game
        num_rounds: Number of rounds
        positions_per_combo: How many position variations to include per player combination
        
    Returns:
        List of matches (agent tuples in play order)
    """
    matches = []
    
    for _ in range(num_rounds):
        # Get all combinations of players
        for combo in itertools.combinations(agents, num_players_per_game):
            # Get limited position permutations
            perms = list(itertools.permutations(combo))
            # Take only the first N permutations
            for perm in perms[:min(positions_per_combo, len(perms))]:
                matches.append(perm)
    
    return matches


def create_vs_baseline_schedule(
    evolved_agents: List[str], 
    baseline_agent: str, 
    num_rounds: int
) -> List[List[str]]:
    """
    Create schedule where each evolved agent plays baseline num_rounds times.
    
    Each pair plays twice per round (positions swapped) for fairness.
    Uses interleaved scheduling for better load distribution.
    Games will still run concurrently like round_robin.
    
    Args:
        evolved_agents: List of evolved agent IDs
        baseline_agent: ID of the baseline agent
        num_rounds: Number of times each evolved agent plays baseline
        
    Returns:
        List of matches where each match is [agent1, agent2]
    """
    schedule = []
    for round_idx in range(num_rounds):
        for evolved_agent in evolved_agents:
            # Each pair plays twice with positions swapped
            schedule.append([evolved_agent, baseline_agent])  # Evolved as player 0
            schedule.append([baseline_agent, evolved_agent])  # Evolved as player 1
    return schedule


def count_games_per_agent(schedule: List[Tuple[str, ...]]) -> dict:
    """
    Count how many games each agent plays in the schedule.
    
    Args:
        schedule: Tournament schedule (list of match tuples)
        
    Returns:
        Dictionary mapping agent names to game counts
    """
    game_count = {}
    
    for match in schedule:
        for agent in match:
            game_count[agent] = game_count.get(agent, 0) + 1
    
    return game_count