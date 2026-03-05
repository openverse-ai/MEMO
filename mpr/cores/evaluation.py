"""
MPR Offline Evaluation System
Professional evaluation framework for MPR agents with trajectory recording.
"""

import os
import json
import asyncio
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

import textarena as ta
from .templates import apply_template, extract_action_and_format_feedback

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for a single game evaluation."""
    env_id: str
    num_players: int
    model_names: List[str]  # List of model names (player1, player2, ...)
    prompt_templates: List[str]  # Template names for each player
    evolved_prompts: Optional[List[str]] = None  # Optional evolved prompts
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: str
    env_id: str
    model_names: List[str]
    prompt_templates: List[str]
    rewards: List[float]
    winner: Optional[int]
    trajectory: List[Dict[str, Any]]
    game_info: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def create_agent(model_name: str, template_name: str, evolved_prompt: str = None) -> ta.core.Agent:
    """Create TextArena agent with specified model and prompt template."""
    
    # Try to use OpenAIOpenrouterAgent if available
    try:
        from textarena.agents import OpenAIOpenrouterAgent
        BaseAgentClass = OpenAIOpenrouterAgent
    except ImportError:
        BaseAgentClass = ta.agents.OpenRouterAgent
    
    # Create a custom agent that applies the template to observations
    class TemplatedOpenRouterAgent(BaseAgentClass):
        def __init__(self, model_name: str, template_name: str, evolved_prompt: str = None, **kwargs):
            super().__init__(model_name=model_name, **kwargs)
            self.template_name = template_name
            self.evolved_prompt = evolved_prompt
            
        def __call__(self, observation: str) -> str:
            # Apply template to observation
            formatted_observation = apply_template(self.template_name, observation, self.evolved_prompt)
            # Call parent with formatted observation
            return super().__call__(formatted_observation)
    
    return TemplatedOpenRouterAgent(
        model_name=model_name,
        template_name=template_name,
        evolved_prompt=evolved_prompt
    )


async def run_single_game(config: GameConfig) -> GameResult:
    """
    Run a single game and return detailed results with trajectory.
    
    Args:
        config: Game configuration
        
    Returns:
        GameResult with trajectory and metadata
    """
    # Generate unique game ID
    game_id = hashlib.md5(
        f"{config.env_id}_{'-'.join(config.model_names)}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    # Create environment
    env = ta.make(config.env_id)
    env.reset(num_players=config.num_players)
    
    # Create agents
    agents = []
    for i, (model_name, template_name) in enumerate(zip(config.model_names, config.prompt_templates)):
        evolved_prompt = config.evolved_prompts[i] if config.evolved_prompts else None
        agent = create_agent(model_name, template_name, evolved_prompt)
        agents.append(agent)
    
    # Track trajectory
    trajectory = []
    done = False
    
    logger.debug(f"Starting game {game_id} with {config.model_names}")
    
    while not done:
        pid, obs = env.get_observation()
        
        # Record state
        step_data = {
            "step": len(trajectory),
            "player_id": pid,
            "observation": obs,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get action from appropriate agent
        try:
            raw_action = agents[pid](obs)
            action, format_feedback = extract_action_and_format_feedback(raw_action)
            
            step_data.update({
                "raw_action": raw_action,
                "action": action,
                "format_feedback": format_feedback
            })
            
        except Exception as e:
            logger.error(f"Agent {pid} error: {e}")
            action = "invalid_action"
            step_data.update({
                "raw_action": str(e),
                "action": action,
                "format_feedback": {"correct_answer_format": False},
                "error": str(e)
            })
        
        # Step environment
        done, step_info = env.step(action=action)
        step_data["step_info"] = step_info
        
        trajectory.append(step_data)
    
    # Get final results
    rewards, game_info = env.close()
    
    # Determine winner
    winner = None
    if len(set(rewards)) > 1:  # Not all equal (tie)
        max_reward = max(rewards)
        winners = [i for i, r in enumerate(rewards) if r == max_reward]
        winner = winners[0] if len(winners) == 1 else None
    
    return GameResult(
        game_id=game_id,
        env_id=config.env_id,
        model_names=config.model_names,
        prompt_templates=config.prompt_templates,
        rewards=rewards,
        winner=winner,
        trajectory=trajectory,
        game_info=game_info,
        timestamp=datetime.now().isoformat(),
        metadata=config.metadata or {}
    )


class MPROfflineEvaluator:
    """
    Professional MPR offline evaluation system.
    
    Features:
        - Async game execution with trajectory recording
        - Flexible model and prompt configuration
        - Comprehensive result analysis
        - CSV and JSON output formats
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "mpr_eval_results",
        max_concurrent_games: int = 4,
        save_trajectories: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory for saving results
            max_concurrent_games: Maximum concurrent games
            save_trajectories: Whether to save detailed trajectories
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_games = max_concurrent_games
        self.save_trajectories = save_trajectories
        
        # Result storage
        self.results: List[GameResult] = []
        
        logger.info(f"MPR Evaluator initialized, output: {self.output_dir}")
    
    async def evaluate_configs(
        self,
        configs: List[GameConfig],
        num_episodes_per_config: int = 8,
        description: str = "MPR Evaluation"
    ) -> pd.DataFrame:
        """
        Evaluate multiple configurations asynchronously.
        
        Args:
            configs: List of game configurations
            num_episodes_per_config: Episodes per configuration
            description: Progress bar description
            
        Returns:
            Summary DataFrame
        """
        # Generate all game configs
        all_games = []
        for config in configs:
            for episode in range(num_episodes_per_config):
                episode_config = GameConfig(
                    env_id=config.env_id,
                    num_players=config.num_players,
                    model_names=config.model_names,
                    prompt_templates=config.prompt_templates,
                    evolved_prompts=config.evolved_prompts,
                    metadata={**(config.metadata or {}), "episode": episode}
                )
                all_games.append(episode_config)
        
        # Run games with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_games)
        
        async def run_with_semaphore(game_config):
            async with semaphore:
                return await run_single_game(game_config)
        
        # Execute all games with progress tracking
        tasks = [run_with_semaphore(config) for config in all_games]
        results = []
        
        # Use tqdm with asyncio.as_completed
        with atqdm(total=len(all_games), desc=description) as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
            
        self.results.extend(results)
        
        # Save detailed results
        if self.save_trajectories:
            await self._save_trajectories(results)
        
        # Generate summary
        summary_df = self._generate_summary(results, configs, num_episodes_per_config)
        await self._save_summary(summary_df)
        
        return summary_df
    
    def _generate_summary(
        self, 
        results: List[GameResult], 
        configs: List[GameConfig],
        num_episodes: int
    ) -> pd.DataFrame:
        """Generate summary statistics for N-player games."""
        summary_data = defaultdict(list)
        
        # Group results by configuration
        config_results = defaultdict(list)
        for result in results:
            config_key = f"{result.env_id}_{'-'.join(result.model_names)}"
            config_results[config_key].append(result)
        
        # Calculate statistics for each configuration
        for config in configs:
            config_key = f"{config.env_id}_{'-'.join(config.model_names)}"
            config_games = config_results[config_key]
            
            if not config_games:
                continue
            
            num_players = len(config.model_names)
            
            # Win statistics for each player
            wins_per_player = [
                sum(1 for r in config_games if r.winner == i) 
                for i in range(num_players)
            ]
            draws = sum(1 for r in config_games if r.winner is None)
            
            # Average rewards per player
            avg_rewards = [
                np.mean([r.rewards[i] if i < len(r.rewards) else 0 for r in config_games])
                for i in range(num_players)
            ]
            
            # Game quality metrics
            invalid_moves_per_player = [
                sum(
                    1 for r in config_games
                    for step in r.trajectory
                    if step.get("player_id") == i and step.get("error")
                ) for i in range(num_players)
            ]
            
            avg_game_length = np.mean([len(r.trajectory) for r in config_games])
            
            # Format feedback per player
            format_stats_per_player = []
            for i in range(num_players):
                correct_formats = sum(
                    1 for r in config_games
                    for step in r.trajectory
                    if (step.get("player_id") == i and 
                        step.get("format_feedback", {}).get("correct_answer_format", False))
                )
                total_moves_player = sum(
                    1 for r in config_games
                    for step in r.trajectory
                    if step.get("player_id") == i
                )
                format_rate = correct_formats / max(total_moves_player, 1)
                format_stats_per_player.append(format_rate)
            
            # Store summary - base fields
            summary_data["env_id"].append(config.env_id)
            summary_data["num_players"].append(num_players)
            summary_data["episodes"].append(len(config_games))
            summary_data["draws"].append(draws)
            summary_data["draw_rate"].append(draws / len(config_games))
            summary_data["avg_game_length"].append(avg_game_length)
            
            # Per-player statistics
            for i in range(num_players):
                summary_data[f"model_p{i}"].append(config.model_names[i])
                summary_data[f"template_p{i}"].append(config.prompt_templates[i])
                summary_data[f"wins_p{i}"].append(wins_per_player[i])
                summary_data[f"win_rate_p{i}"].append(wins_per_player[i] / len(config_games))
                summary_data[f"avg_reward_p{i}"].append(avg_rewards[i])
                summary_data[f"invalid_moves_p{i}"].append(invalid_moves_per_player[i])
                summary_data[f"format_accuracy_p{i}"].append(format_stats_per_player[i])
            
            # Fill remaining player slots with N/A for consistent DataFrame structure
            max_players = max(len(config.model_names) for config in configs)
            for i in range(num_players, max_players):
                for field in [f"model_p{i}", f"template_p{i}"]:
                    if field not in summary_data:
                        summary_data[field] = []
                    summary_data[field].append("N/A")
                for field in [f"wins_p{i}", f"win_rate_p{i}", f"avg_reward_p{i}", 
                             f"invalid_moves_p{i}", f"format_accuracy_p{i}"]:
                    if field not in summary_data:
                        summary_data[field] = []
                    summary_data[field].append(0.0)
        
        return pd.DataFrame(summary_data)
    
    async def _save_trajectories(self, results: List[GameResult]):
        """Save detailed game trajectories."""
        trajectories_dir = self.output_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        
        for result in results:
            trajectory_file = trajectories_dir / f"{result.game_id}.json"
            with open(trajectory_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
    
    async def _save_summary(self, summary_df: pd.DataFrame):
        """Save summary statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_file = self.output_dir / f"mpr_eval_summary_{timestamp}.csv"
        summary_df.to_csv(csv_file, index=False)
        
        # Pretty print to console
        print("\n=== MPR Evaluation Summary ===")
        print(summary_df.to_markdown(index=False, floatfmt=".3f"))
        print(f"\nResults saved to: {self.output_dir}")
        
        logger.info(f"Summary saved to {csv_file}")


# Convenience function for simple evaluations
# async def evaluate_models_simple(
#     env_ids: List[str],
#     model_configs: List[Dict[str, Any]],
#     num_episodes: int = 8,
#     output_dir: str = "mpr_eval_results"
# ) -> pd.DataFrame:
#     """
#     Simple evaluation interface.
    
#     Args:
#         env_ids: List of environment IDs
#         model_configs: List of {"models": [...], "templates": [...]} dicts
#         num_episodes: Episodes per configuration
#         output_dir: Output directory
        
#     Returns:
#         Summary DataFrame
#     """
#     evaluator = MPROfflineEvaluator(output_dir=output_dir)
    
#     configs = []
#     for env_id in env_ids:
#         env = ta.make(env_id)
#         num_players = getattr(env, 'num_players', None)
#         env.reset(num_players=num_players)
#         # Try to get num_players from environment, fallback to model count
        
        
#         for model_config in model_configs:
#             # Use env num_players if available, otherwise use model count
#             actual_num_players = num_players if num_players is not None else len(model_config["models"])
            
#             # Validate model count matches expected players
#             if num_players is not None and len(model_config["models"]) != num_players:
#                 logger.warning(f"Environment {env_id} expects {num_players} players, "
#                              f"but got {len(model_config['models'])} models. Using {len(model_config['models'])}")
#                 actual_num_players = len(model_config["models"])
            
#             config = GameConfig(
#                 env_id=env_id,
#                 num_players=actual_num_players,
#                 model_names=model_config["models"],
#                 prompt_templates=model_config["templates"],
#                 evolved_prompts=model_config.get("evolved_prompts"),
#                 metadata=model_config.get("metadata", {})
#             )
#             configs.append(config)
    
#     return await evaluator.evaluate_configs(configs, num_episodes)