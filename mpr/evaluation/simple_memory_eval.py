#!/usr/bin/env python3
"""
Memory-Enhanced MPR Offline Evaluation with Fair Round-Robin Tournament

Usage:
    python simple_memory_eval.py "google/gemini-2.0-flash-001" "Qwen/Qwen3-4B" --memory-path "./insights"
    python simple_memory_eval.py "model1" "model2" --memory-agents 0 1 --memory-path "./generation_5_insights"
    python simple_memory_eval.py "model1" "model2" "model3" --memory-agents 0 2 --memory-templates "default" "adaptive"
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

# Reduce httpx logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)

import textarena as ta
from mpr.cores.game_runner import run_single_game, GameInformation
from mpr.cores.tournament_scheduler import create_round_robin_schedule, create_vs_baseline_schedule, count_games_per_agent
from mpr.tournament.agent_pool import AgentPool
from mpr.replaybuffer.replaybuffer import ReplayBuffer

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Memory-Enhanced Fair Round-Robin Tournament Evaluation")
    
    # Required models list
    parser.add_argument("models", nargs='+', help="List of models to compete")
    
    # Tournament settings
    parser.add_argument("--num-players", type=int, default=2, help="Number of players per game (default: 2)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of tournament rounds (default: 5)")
    parser.add_argument("--env", default="CustomTicTacToe-v1", help="Environment (default: CustomTicTacToe-v1)")
    
    # Memory settings
    parser.add_argument("--memory-path", required=True, help="Path to memory/insights directory")
    parser.add_argument("--memory-agents", nargs='*', type=int, help="Indices of agents to enhance with memory (default: all)")
    parser.add_argument("--memory-templates", nargs='*', help="Templates for memory-enhanced agents")
    
    # Optional parameters
    parser.add_argument("--output", default="memory_tournament_results", help="Output directory")
    parser.add_argument("--templates", nargs='*', help="Templates for models")
    parser.add_argument("--evolved-prompts", nargs='*', help="Evolved prompts for models")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent games")
    parser.add_argument("--save-trajectories", action="store_true", help="Save detailed trajectories")
    
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


def run_tournament(agent_pool: AgentPool, env_id: str, num_players_per_game: int, 
                  num_rounds: int, max_concurrent: int, output_dir: str,
                  save_trajectories: bool = True, memory_agents: List[int] = None,
                  generation: int = None, phase: str = "tournament", 
                  format_requirement_level: str = "strict", schedule_type: str = "robin",
                  baseline_model: str = None, replay_buffer: Optional[ReplayBuffer] = None, temperature: float = None,
                  track_tokens: bool = False) -> Dict:
    """Run tournament with memory-enhanced agents.
    
    Args:
        agent_pool: Pool of agents to compete
        env_id: Environment ID for the games
        num_players_per_game: Number of players per game
        num_rounds: Number of tournament rounds
        max_concurrent: Maximum concurrent games
        output_dir: Directory to save results
        save_trajectories: Whether to save game trajectories
        memory_agents: List of agent indices to enhance with memory
        generation: Generation number for naming
        phase: Tournament phase ("tournament" or "evaluation")
        format_requirement_level: How strict to be about format requirements
        schedule_type: "robin" for round-robin, "vs_baseline" for baseline comparison, or "vs_best" for best candidate comparison
        baseline_model: Model to use for baseline agent in vs_baseline mode, or None to use first agent as baseline in vs_best mode
    """
    
    # Get agent names
    agent_names = list(agent_pool.agents.keys())
    logger.info(f"Tournament with {len(agent_names)} agents: {agent_names}")
    
    # Create baseline agent if needed
    baseline_id = None
    if schedule_type == "vs_baseline":
        baseline_id = "baseline_agent"
        # Add baseline agent to pool
        # Use provided baseline_model or fall back to same model as evolved agents
        baseline_model_name = baseline_model
        agent_pool.add_agent(
            agent_id=baseline_id,
            model_name=baseline_model_name,
            template_name="gemini-boxed",  # Fixed template for baseline
            evolved_prompt=None,  # No evolved prompt
            temperature=temperature
        )
        # Update agent_names to include baseline
        agent_names = list(agent_pool.agents.keys())
        logger.info(f"Added baseline agent ({baseline_model_name}) for vs_baseline schedule")
    
    # Create schedule based on schedule_type
    if schedule_type == "robin":
        schedule = create_round_robin_schedule(agent_names, num_players_per_game, num_rounds)
    elif schedule_type == "vs_baseline":
        # Get evolved agents (all except baseline)
        evolved_agents = [name for name in agent_names if name != baseline_id]
        schedule = create_vs_baseline_schedule(evolved_agents, baseline_id, num_rounds)
    elif schedule_type == "vs_best":
        # Use first agent as the "best" agent to compare against
        # IMPORTANT: The caller must ensure the best agent is added to the pool FIRST
        best_agent_id = agent_names[0]
        other_agents = agent_names[1:]
        schedule = create_vs_baseline_schedule(other_agents, best_agent_id, num_rounds)
        logger.info(f"vs_best schedule: {best_agent_id} will play against {len(other_agents)} other agents")
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")
    
    games_per_agent = count_games_per_agent(schedule)
    
    logger.info(f"Generated {len(schedule)} games over {num_rounds} rounds")
    logger.info("Games per agent: " + ", ".join([f"{agent}: {count}" for agent, count in games_per_agent.items()]))
    
    # Run games with threading
    results = []
    trajectories = []
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all games
        future_to_game = {}
        for i, agent_order in enumerate(schedule):

            # decide whether to replay actions
            rand = random.random()
            # check if the replay buffer is being used
            # logger.info(f"REPLAY BUFFER BETA: {replay_buffer.beta if replay_buffer is not None else 'N/A'}")
            use_replay = (replay_buffer is not None and rand < replay_buffer.beta and len(replay_buffer) > 0)
            if replay_buffer is not None and getattr(replay_buffer, 'debug', False):
                logger.info(f"Rand: {rand}, Beta: {replay_buffer.beta}, len(replayBuffer): {len(replay_buffer)}")
            if use_replay:
                sampled_replay_episode = replay_buffer.sample_prioritized(batch_size=1)[0]
                replay_actions = sampled_replay_episode["played_actions"]
                seed = sampled_replay_episode["seed"]
                if replay_buffer is not None and getattr(replay_buffer, 'debug', False):
                    logger.info(f"    - Using replay for game {i} with seed value {seed}")
                    logger.info(f"    - Replay episode from game that looks like:\n{replay_actions}")
            else:
                replay_actions = None
                seed = random.randint(0, 1000)
                # logger.info(f"    - No replay for game {i}")
            
            future = executor.submit(
                run_single_game,
                agents=agent_pool.agents,
                env_id=env_id,
                agent_order=agent_order,
                metadata={
                    "match_idx": i,
                    "replay_actions": replay_actions,
                    "replay_seed": seed,
                },
                format_requirement_level=format_requirement_level,
                track_tokens=track_tokens
            )
            future_to_game[future] = (i, agent_order)
        
        # Collect results with progress bar
        pbar = tqdm(total=len(schedule), desc="Running games")
        for future in as_completed(future_to_game):
            game_index, agent_order = future_to_game[future]
            
            # Update progress bar description to show completed matchup
            matchup = " vs ".join(agent_order)
            pbar.set_description(f"Just finished: {matchup}")
            
            game_result = future.result()
            results.append(game_result)
            pbar.update(1)
            
            if save_trajectories:
                trajectories.append({
                    "game_id": game_result.game_id,
                    "agent_names": game_result.agent_names,
                    "rewards": game_result.rewards,
                    "trajectory": game_result.trajectory
                })
        
        pbar.close()

    
    # Process results and update agent pool performance
    match_results_for_trueskill = []
    for result in results:
        if result.used_replay: # Skip games with the use of replay.
            logger.info(f"Skipping game {result.game_id} from stats due to use of replay.")
            continue
        else:
            # Store match with agents and rewards for TrueSkill
            match_results_for_trueskill.append({
                'agents': result.agent_names,
                'rewards': result.rewards
            })
            
            # Update win/loss/draw counts
            for i, agent_name in enumerate(result.agent_names):
                # Skip baseline performance tracking in vs_baseline mode
                if schedule_type == "vs_baseline" and agent_name == baseline_id:
                    continue
                
                reward = result.rewards[i]
                
                if reward == 1:
                    agent_pool.update_game_result(agent_name, "win")
                elif reward == -1:
                    agent_pool.update_game_result(agent_name, "loss")
                else:  # reward == 0
                    agent_pool.update_game_result(agent_name, "draw")
    
    # Batch update TrueSkill ratings after all games complete
    agent_pool.batch_update_trueskill(match_results_for_trueskill)
    
    # Clean up baseline agent if needed
    if schedule_type == "vs_baseline" and baseline_id:
        # Remove baseline from pool before any further processing
        agent_pool.delete_agent(baseline_id)
        # Also remove baseline from agent_names for stats computation
        agent_names = [name for name in agent_names if name != baseline_id]
    
    # Compute statistics
    stats = compute_tournament_stats(results, agent_names)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which agents have memory enhancement
    memory_enhanced_agents = []
    if memory_agents:
        memory_enhanced_agents = [agent_names[i] for i in memory_agents if i < len(agent_names)]
    
    # Save summary file with informative naming
    if generation is not None:
        summary_filename = f"summary_gen{generation}_{phase}.json"
    else:
        summary_filename = f"summary_{phase}_{timestamp}.json"
    
    summary_file = output_path / summary_filename
    summary_data = {
        "env_id": env_id,
        "num_agents": len(agent_names),
        "num_players_per_game": num_players_per_game,
        "num_rounds": num_rounds,
        "total_games": len(results),
        "agent_stats": {name: stats["agents"][name] for name in agent_names},
        "memory_enhanced_agents": memory_enhanced_agents,
        "timestamp": timestamp
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Tournament summary saved silently
    
    # Save trajectories if requested with informative naming
    if save_trajectories:
        if generation is not None:
            traj_filename = f"trajectories_gen{generation}_{phase}.json"
        else:
            traj_filename = f"trajectories_{phase}_{timestamp}.json"
            
        traj_file = output_path / traj_filename
        traj_data = [result.to_dict() for result in results]
        with open(traj_file, 'w') as f:
            json.dump(traj_data, f, indent=2)
        # Trajectories saved silently

    if replay_buffer is not None:
        added = sum(replay_buffer.push_batch(result) for result in results)
        logger.info(f"[Tournament] Added {added} new states → ReplayBuffer size: {len(replay_buffer)}")
    
    return stats, results


def compute_tournament_stats(results: List[GameInformation], agent_names: List[str]) -> Dict:
    """Compute tournament statistics."""
    stats = {
        "games_played": len(results),
        "agents": {}
    }
    
    # Initialize agent stats
    for agent in agent_names:
        stats["agents"][agent] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_reward": 0.0,
            "win_rate": 0.0,
            "total_format_errors": 0,
            "total_invalid_moves": 0,
            "avg_turns_per_game": 0.0
        }
    
    # Aggregate results
    for result in results:
        for i, agent_name in enumerate(result.agent_names):
            # Skip agents not in our agent_names list (e.g., baseline agent)
            if agent_name not in stats["agents"]:
                continue
                
            agent_stats = stats["agents"][agent_name]
            agent_stats["games_played"] += 1
            agent_stats["total_reward"] += result.rewards[i]
            agent_stats["total_format_errors"] += result.format_errors_per_agent.get(agent_name, 0)
            agent_stats["total_invalid_moves"] += result.invalid_moves_per_agent.get(agent_name, 0)
            agent_stats["avg_turns_per_game"] += result.num_turns
            
            # Count wins/losses/draws
            reward = result.rewards[i]
            if reward > 0:
                agent_stats["wins"] += 1
            elif reward < 0:
                agent_stats["losses"] += 1
            else:
                agent_stats["draws"] += 1
    
    # Calculate final metrics
    for agent_name, agent_stats in stats["agents"].items():
        if agent_stats["games_played"] > 0:
            agent_stats["win_rate"] = agent_stats["wins"] / agent_stats["games_played"]
            agent_stats["avg_turns_per_game"] /= agent_stats["games_played"]
    
    return stats


def print_tournament_summary(stats: Dict):
    """Print tournament summary."""
    # Print final standings
    print("\n" + "="*60)
    print("TOURNAMENT FINAL STANDINGS")
    print("="*60)
    
    # Sort agents by wins and avg_reward
    sorted_agents = sorted(stats["agents"].items(), key=lambda x: (x[1]['wins'], x[1]['total_reward']/x[1]['games_played'] if x[1]['games_played'] > 0 else 0), reverse=True)
    
    for rank, (agent_name, agent_stats) in enumerate(sorted_agents, 1):
        print(f"{rank}. {agent_name}:")
        print(f"   Wins: {agent_stats['wins']}/{agent_stats['games_played']} ({agent_stats['win_rate']:.1%})")
        avg_reward = agent_stats['total_reward'] / agent_stats['games_played'] if agent_stats['games_played'] > 0 else 0
        print(f"   Avg Reward: {avg_reward:.3f}")
        if agent_stats.get('total_invalid_moves', 0) > 0 or agent_stats.get('total_format_errors', 0) > 0:
            print(f"   Errors: {agent_stats.get('total_invalid_moves', 0)} invalid moves, {agent_stats.get('total_format_errors', 0)} format errors")


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("MEMORY-ENHANCED TOURNAMENT EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Models: {args.models}")
    logger.info(f"Memory path: {args.memory_path}")
    logger.info(f"Memory agents: {args.memory_agents or 'all'}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Players per game: {args.num_players}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info("=" * 60)
    
    # Create agent pool with memory enhancement
    agent_pool = AgentPool()
    agent_pool.create_memory_agents_from_models(
        model_names=args.models,
        memory_path=args.memory_path,
        memory_agents=args.memory_agents,
        templates=args.templates,
        evolved_prompts=args.evolved_prompts
    )
    
    # Run tournament
    stats = run_tournament(
        agent_pool=agent_pool,
        env_id=args.env,
        num_players_per_game=args.num_players,
        num_rounds=args.rounds,
        max_concurrent=args.max_concurrent,
        output_dir=args.output,
        save_trajectories=args.save_trajectories,
        memory_agents=args.memory_agents
    )
    
    # Print summary
    print_tournament_summary(stats)
    
    logger.info(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()