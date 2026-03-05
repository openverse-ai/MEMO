#!/usr/bin/env python3
"""
Simple MPR Offline Evaluation with Fair Round-Robin Tournament

Usage:
    python simple_eval.py "google/gemini-2.0-flash-001" "Qwen/Qwen3-4B"
    python simple_eval.py "google/gemini-2.0-flash-001" "Qwen/Qwen3-4B" --rounds 5
    python simple_eval.py "model1" "model2" "model3" --num-players 2 --rounds 3
    python simple_eval.py "model1" "model2" "model3" "model4" --num-players 3 --env "CustomTicTacToe-v1"
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Reduce httpx logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)

import textarena as ta
from mpr.cores.game_runner import run_single_game, GameInformation
from mpr.cores.tournament_scheduler import create_round_robin_schedule, count_games_per_agent
from mpr.tournament.agent_pool import AgentPool

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fair Round-Robin Tournament Evaluation")
    
    # Required models list
    parser.add_argument("models", nargs='+', help="List of models to compete")
    
    # Tournament settings
    parser.add_argument("--num-players", type=int, default=2, help="Number of players per game (default: 2)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of tournament rounds (default: 5)")
    parser.add_argument("--env", default="CustomTicTacToe-v1", help="Environment (default: CustomTicTacToe-v1)")
    
    # Optional parameters
    parser.add_argument("--output", default="tournament_results", help="Output directory")
    parser.add_argument("--templates", nargs='*', help="Templates for models")
    parser.add_argument("--evolved-prompts", nargs='*', help="Evolved prompts for models")
    parser.add_argument("--concurrent", type=int, default=50, help="Max concurrent games")
    parser.add_argument("--save-trajectories", action="store_true", help="Save game trajectories")
    
    return parser.parse_args()


def create_agent_pool(
    models: List[str],
    templates: List[str] = None,
    evolved_prompts: List[str] = None
) -> AgentPool:
    """Create agent pool from model names with template support."""
    pool = AgentPool()
    
    # Create agents with templates and evolved prompts
    pool.create_agents_from_models(models, templates, evolved_prompts)
    
    logger.info(f"Created agent pool with {len(models)} agents")
    
    return pool


def run_tournament(
    agent_pool: AgentPool,
    env_id: str,
    num_players_per_game: int,
    num_rounds: int,
    max_concurrent: int,
    output_dir: str,
    save_trajectories: bool = False
):
    """Run a fair round-robin tournament."""
    # Get agent dictionary from pool
    agents = agent_pool.agents
    agent_names = agent_pool.get_all_agent_ids()
    
    # Create tournament schedule
    logger.info(f"Creating tournament schedule: {len(agent_names)} agents, {num_players_per_game} players per game, {num_rounds} rounds")
    schedule = create_round_robin_schedule(agent_names, num_players_per_game, num_rounds)
    logger.info(f"Total games to play: {len(schedule)}")
    
    # Count games per agent for fairness check
    game_counts = count_games_per_agent(schedule)
    for agent, count in game_counts.items():
        logger.info(f"  {agent}: {count} games")
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run all games concurrently
    results = []
    wins = defaultdict(int)
    losses = defaultdict(int)
    draws = defaultdict(int)
    total_rewards = defaultdict(float)
    match_results_for_trueskill = []  # Track for TrueSkill update
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all games
        futures = []
        for match_idx, agent_order in enumerate(schedule):
            future = executor.submit(
                run_single_game,
                agents=agents,
                env_id=env_id,
                agent_order=agent_order,
                metadata={"match_idx": match_idx}
            )
            futures.append(future)
        
        # Collect results (just gather them during concurrent execution)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running games"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.warning(f"Game failed: {e}")
    
    # Process all results after tournament completes
    logger.info("Processing tournament results...")
    
    # Prepare match results for TrueSkill batch update
    for result in results:
        # Store match with agents and rewards for TrueSkill
        match_results_for_trueskill.append({
            'agents': result.agent_names,
            'rewards': result.rewards
        })
        
        # Update win/loss/draw counts based on rewards
        for i, agent_name in enumerate(result.agent_names):
            reward = result.rewards[i]
            
            if reward == 1:
                wins[agent_name] += 1
                agent_pool.update_game_result(agent_name, "win")
            elif reward == -1:
                losses[agent_name] += 1
                agent_pool.update_game_result(agent_name, "loss")
            else:  # reward == 0
                draws[agent_name] += 1
                agent_pool.update_game_result(agent_name, "draw")
            
            # Track total rewards
            total_rewards[agent_name] += reward
    
    # Batch update TrueSkill ratings after all games complete
    agent_pool.batch_update_trueskill(match_results_for_trueskill)
    
    # Calculate final statistics with error tracking
    stats = {}
    total_invalid = defaultdict(int)
    total_format_errors = defaultdict(int)
    
    # Aggregate error stats
    for result in results:
        for agent_name, count in result.invalid_moves_per_agent.items():
            total_invalid[agent_name] += count
        for agent_name, count in result.format_errors_per_agent.items():
            total_format_errors[agent_name] += count
    
    # Create final stats including TrueSkill ratings
    for agent_name in agent_names:
        games_played = game_counts[agent_name]
        perf = agent_pool.get_agent_performance(agent_name)
        
        stats[agent_name] = {
            "games_played": games_played,
            "wins": wins[agent_name],
            "losses": losses[agent_name],
            "draws": draws[agent_name],
            "win_rate": wins[agent_name] / games_played if games_played > 0 else 0,
            "total_reward": total_rewards[agent_name],
            "avg_reward": total_rewards[agent_name] / games_played if games_played > 0 else 0,
            "trueskill_mu": perf.trueskill_rating.mu if perf else -1,
            "trueskill_sigma": perf.trueskill_rating.sigma if perf else -1,
            "invalid_moves": total_invalid[agent_name],
            "format_errors": total_format_errors[agent_name]
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_file = output_path / f"tournament_summary_{timestamp}.json"
    summary_data = {
        "env_id": env_id,
        "num_agents": len(agent_names),
        "num_players_per_game": num_players_per_game,
        "num_rounds": num_rounds,
        "total_games": len(results),
        "agent_stats": stats,
        "timestamp": timestamp
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"\nTournament Summary saved to {summary_file}")
    
    # Save trajectories if requested
    if save_trajectories:
        traj_file = output_path / f"trajectories_{timestamp}.json"
        traj_data = [result.to_dict() for result in results]
        with open(traj_file, 'w') as f:
            json.dump(traj_data, f, indent=2)
        logger.info(f"Trajectories saved to {traj_file}")
    
    # Print final standings
    print("\n" + "="*60)
    print("TOURNAMENT FINAL STANDINGS")
    print("="*60)
    
    sorted_agents = sorted(stats.items(), key=lambda x: (x[1]['wins'], x[1]['avg_reward']), reverse=True)
    for rank, (agent_name, agent_stats) in enumerate(sorted_agents, 1):
        print(f"{rank}. {agent_name}:")
        print(f"   Wins: {agent_stats['wins']}/{agent_stats['games_played']} ({agent_stats['win_rate']:.1%})")
        print(f"   Avg Reward: {agent_stats['avg_reward']:.3f}")
        if agent_stats.get('invalid_moves', 0) > 0 or agent_stats.get('format_errors', 0) > 0:
            print(f"   Errors: {agent_stats.get('invalid_moves', 0)} invalid moves, {agent_stats.get('format_errors', 0)} format errors")
    
    return stats


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate inputs
    if len(args.models) < 2:
        logger.error("Need at least 2 models")
        return
    
    if args.num_players > len(args.models):
        logger.error(f"Number of players per game ({args.num_players}) cannot exceed number of models ({len(args.models)})")
        return
    
    # Process templates
    templates = args.templates if args.templates else [None] * len(args.models)
    if args.templates and len(args.templates) != len(args.models):
        logger.error(f"Template count ({len(args.templates)}) must match model count ({len(args.models)})")
        return
    
    # Process evolved prompts
    evolved_prompts = args.evolved_prompts if args.evolved_prompts else [None] * len(args.models)
    if args.evolved_prompts and len(args.evolved_prompts) != len(args.models):
        logger.error(f"Evolved prompts count ({len(args.evolved_prompts)}) must match model count ({len(args.models)})")
        return
    
    logger.info(f"Starting tournament: {len(args.models)} models, {args.num_players} players per game, {args.rounds} rounds")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Models: {', '.join(args.models)}")
    
    try:
        # Create agent pool
        agent_pool = create_agent_pool(args.models, templates, evolved_prompts)
        
        # Run tournament
        stats = run_tournament(
            agent_pool=agent_pool,
            env_id=args.env,
            num_players_per_game=args.num_players,
            num_rounds=args.rounds,
            max_concurrent=args.concurrent,
            output_dir=args.output,
            save_trajectories=args.save_trajectories
        )
        
        logger.info("✅ Tournament completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Tournament failed: {e}")
        raise


if __name__ == "__main__":
    main()