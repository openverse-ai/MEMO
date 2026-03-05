"""
MPR Offline Evaluation System - ThreadPoolExecutor Version
Simpler, faster development with familiar thread-based concurrency.
"""

import os
import json
import logging
import hashlib
from pathlib import Path

# Reduce httpx logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import textarena as ta
from .templates import apply_template, extract_action_and_format_feedback

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for a single game evaluation."""
    env_id: str
    num_players: int
    model_names: List[str]
    prompt_templates: List[str]
    evolved_prompts: Optional[List[str]] = None
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
    
    # Create a custom agent that applies the template to observations
    class TemplatedOpenRouterAgent(ta.agents.OpenRouterAgent):
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


def run_single_game(config: GameConfig) -> GameResult:
    """
    Run a single game and return detailed results with trajectory.
    SIMPLE SYNC FUNCTION - No async complexity!
    """
    # Generate unique game ID
    game_id = hashlib.md5(
        f"{config.env_id}_{'-'.join(config.model_names)}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    try:
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
                logger.error(f"Agent {pid} error in game {game_id}: {e}")
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
        
        # Determine winner from rewards correctly
        winner = None
        rewards_list = []
        
        if isinstance(rewards, dict):
            # Handle dict rewards (TextArena format)
            winners = [int(pid) for pid, r in rewards.items() if r > 0]
            losers = [int(pid) for pid, r in rewards.items() if r < 0]
            is_draw = all(r == 0 for r in rewards.values())
            
            # Convert to list for storage - handle both string and int keys
            rewards_list = []
            for i in range(config.num_players):
                # Try both string and int keys
                reward = rewards.get(str(i), rewards.get(i, 0))
                rewards_list.append(reward)
            
            # Set winner
            if winners and len(winners) == 1:
                winner = winners[0]
            elif is_draw:
                winner = None
        else:
            # Handle list rewards
            rewards_list = list(rewards)
            winners = [i for i, r in enumerate(rewards_list) if r > 0]
            
            if winners and len(winners) == 1:
                winner = winners[0]
            elif all(r == 0 for r in rewards_list):
                winner = None
        
        return GameResult(
            game_id=game_id,
            env_id=config.env_id,
            model_names=config.model_names,
            prompt_templates=config.prompt_templates,
            rewards=rewards_list,  # Store as list for consistency
            winner=winner,
            trajectory=trajectory,
            game_info=game_info,
            timestamp=datetime.now().isoformat(),
            metadata=config.metadata or {}
        )
        
    except Exception as e:
        logger.error(f"Game {game_id} failed: {e}")
        # Return failed game result
        return GameResult(
            game_id=game_id,
            env_id=config.env_id,
            model_names=config.model_names,
            prompt_templates=config.prompt_templates,
            rewards=[0.0] * config.num_players,
            winner=None,
            trajectory=[{"error": str(e), "timestamp": datetime.now().isoformat()}],
            game_info={"error": str(e)},
            timestamp=datetime.now().isoformat(),
            metadata=config.metadata or {}
        )


class MPROfflineEvaluator:
    """
    Simple threaded MPR evaluation system.
    Much easier to develop and debug than async version!
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "mpr_eval_results",
        max_concurrent_games: int = 16,
        save_trajectories: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_games = max_concurrent_games
        self.save_trajectories = save_trajectories
        self.results: List[GameResult] = []
        
        logger.info(f"MPR Evaluator initialized, output: {self.output_dir}")
    
    def evaluate_configs(
        self,
        configs: List[GameConfig],
        num_episodes_per_config: int = 8,
        description: str = "MPR Evaluation"
    ) -> pd.DataFrame:
        """
        Evaluate multiple configurations.
        SIMPLE SYNC METHOD - No async headaches!
        """
        # Generate all game configs with rotating positions for fairness
        all_games = []
        for config in configs:
            for episode in range(num_episodes_per_config):
                # Rotate player positions: each model gets to be first player
                rotation = episode % config.num_players
                
                # Apply rotation to all arrays
                rotated_models = config.model_names[rotation:] + config.model_names[:rotation]
                rotated_templates = config.prompt_templates[rotation:] + config.prompt_templates[:rotation]
                rotated_prompts = None
                if config.evolved_prompts:
                    rotated_prompts = config.evolved_prompts[rotation:] + config.evolved_prompts[:rotation]
                
                episode_config = GameConfig(
                    env_id=config.env_id,
                    num_players=config.num_players,
                    model_names=rotated_models,
                    prompt_templates=rotated_templates,
                    evolved_prompts=rotated_prompts,
                    metadata={
                        **(config.metadata or {}), 
                        "episode": episode, 
                        "position_rotation": rotation,
                        "original_model_order": config.model_names
                    }
                )
                all_games.append(episode_config)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent_games) as executor:
            # Submit all games
            future_to_config = {
                executor.submit(run_single_game, config): config 
                for config in all_games
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_config), 
                             desc=description, total=len(all_games)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    config = future_to_config[future]
                    logger.error(f"Game failed: {e}")
                    # Could create a failed result here if needed
        
        self.results.extend(results)
        
        # Save detailed results
        if self.save_trajectories:
            self._save_trajectories(results)
        
        # Generate summary
        summary_df = self._generate_summary(results, configs, num_episodes_per_config)
        self._save_summary(summary_df)
        
        return summary_df
    
    def _generate_summary(self, results: List[GameResult], configs: List[GameConfig], num_episodes: int) -> pd.DataFrame:
        """Generate summary statistics for N-player games with position rotation."""
        summary_data = defaultdict(list)
        
        # Group results by original configuration (using original model order)
        config_results = defaultdict(list)
        for result in results:
            # Use original model order for consistent grouping
            original_order = result.metadata.get('original_model_order', result.model_names)
            config_key = f"{result.env_id}_{'-'.join(original_order)}"
            config_results[config_key].append(result)
        
        # Calculate statistics for each configuration
        for config in configs:
            config_key = f"{config.env_id}_{'-'.join(config.model_names)}"
            config_games = config_results[config_key]
            
            if not config_games:
                continue
            
            num_players = len(config.model_names)
            original_model_names = config.model_names  # Original order for display
            
            # Calculate wins and rewards per original model (accounting for rotations)
            wins_per_model = [0] * num_players
            rewards_per_model = [[] for _ in range(num_players)]
            
            for result in config_games:
                # Get the rotation used in this game
                rotation = result.metadata.get('position_rotation', 0)
                
                if result.winner is not None:
                    # Map game winner back to original model index
                    original_winner = (result.winner - rotation) % num_players
                    wins_per_model[original_winner] += 1
                
                # Collect rewards for each original model
                for original_idx in range(num_players):
                    # Map original model index to game position
                    game_position = (original_idx + rotation) % num_players
                    
                    # Rewards should now always be lists after our fix
                    if isinstance(result.rewards, list) and game_position < len(result.rewards):
                        reward_value = result.rewards[game_position]
                    elif isinstance(result.rewards, dict):
                        # Fallback for old data
                        reward_value = result.rewards.get(str(game_position), 0)
                    else:
                        reward_value = 0
                    
                    rewards_per_model[original_idx].append(reward_value)
            
            draws = sum(1 for r in config_games if r.winner is None)
            
            # Average rewards per original model
            avg_rewards = [
                np.mean(rewards) if rewards else 0.0
                for rewards in rewards_per_model
            ]
            
            # Game quality metrics
            avg_game_length = np.mean([len(r.trajectory) for r in config_games])
            
            # Store summary - base fields
            summary_data["env_id"].append(config.env_id)
            summary_data["num_players"].append(num_players)
            summary_data["episodes"].append(len(config_games))
            summary_data["draws"].append(draws)
            summary_data["draw_rate"].append(draws / len(config_games))
            summary_data["avg_game_length"].append(avg_game_length)
            
            # Per-player statistics (using original model order and corrected wins)
            for i in range(num_players):
                summary_data[f"model_p{i}"].append(original_model_names[i])
                summary_data[f"template_p{i}"].append(config.prompt_templates[i])
                summary_data[f"wins_p{i}"].append(wins_per_model[i])
                summary_data[f"win_rate_p{i}"].append(wins_per_model[i] / len(config_games))
                summary_data[f"avg_reward_p{i}"].append(avg_rewards[i])
        
        return pd.DataFrame(summary_data)
    
    def _save_trajectories(self, results: List[GameResult]):
        """Save detailed game trajectories."""
        trajectories_dir = self.output_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        
        for result in results:
            trajectory_file = trajectories_dir / f"{result.game_id}.json"
            with open(trajectory_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
    
    def _save_summary(self, summary_df: pd.DataFrame):
        """Save summary statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = self.output_dir / f"mpr_eval_summary_{timestamp}.json"
        summary_dict = summary_df.to_dict(orient='records')
        with open(json_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        # Save CSV as well
        csv_file = self.output_dir / f"mpr_eval_summary_{timestamp}.csv"
        summary_df.to_csv(csv_file, index=False)
        
        # Pretty print to console (simple format)
        print("\n=== MPR Evaluation Summary ===")
        print(f"Results: {len(summary_dict)} configurations")
        for i, row in enumerate(summary_dict):
            print(f"\nConfig {i+1}:")
            print(f"  Environment: {row.get('env_id', 'N/A')}")
            print(f"  Players: {row.get('num_players', 'N/A')}")
            print(f"  Episodes: {row.get('episodes', 'N/A')}")
            
            # Show per-player stats
            num_players = row.get('num_players', 0)
            for p in range(num_players):
                model = row.get(f'model_p{p}', 'N/A')
                wins = row.get(f'wins_p{p}', 0)
                win_rate = row.get(f'win_rate_p{p}', 0)
                avg_reward = row.get(f'avg_reward_p{p}', 0)
                print(f"  Player {p} ({model}): {wins} wins ({win_rate:.2f} rate, {avg_reward:.2f} avg reward)")
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"JSON: {json_file}")
        print(f"CSV: {csv_file}")
        
        logger.info(f"Summary saved to {json_file} and {csv_file}")