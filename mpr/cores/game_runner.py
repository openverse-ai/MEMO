"""
Game Runner - Runs games with pre-created agents
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import copy

import textarena as ta
from .templates import extract_action_and_format_feedback

logger = logging.getLogger(__name__)


@dataclass
class GameInformation:
    """Complete information about a game."""
    game_id: str
    env_id: str
    agent_names: List[str]
    rewards: List[float]
    winners: List[int]  # List of winner indices (can be multiple for draws)
    trajectory: List[Dict[str, Any]]
    game_info: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any]
    num_turns: int
    used_replay: bool
    invalid_moves_per_agent: Dict[str, int] = field(default_factory=dict)
    format_errors_per_agent: Dict[str, int] = field(default_factory=dict)
    losers: Optional[List[int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with safe serialization of complex objects."""
        result = {}
        
        # Handle basic fields
        for field_name in ["game_id", "env_id", "agent_names", "rewards", "winners", 
                          "timestamp", "num_turns", "used_replay", "invalid_moves_per_agent", 
                          "format_errors_per_agent"]:
            result[field_name] = getattr(self, field_name)
        
        # Handle losers (can be None)
        result["losers"] = self.losers
        
        # Handle metadata (ensure it's serializable)
        result["metadata"] = self._serialize_dict(self.metadata)
        
        # Handle game_info (ensure it's serializable)
        result["game_info"] = self._serialize_dict(self.game_info)
        
        # Handle trajectory (ensure all step data is serializable)
        result["trajectory"] = []
        for step in self.trajectory:
            serialized_step = {}
            for key, value in step.items():
                if key in ["step", "player_id", "agent_name", "raw_action", "action", "seed", "replayed_action"]:
                    # These should be basic types
                    serialized_step[key] = value
                elif key == "observation":
                    # Observation should be a string but ensure it's serializable
                    serialized_step[key] = str(value)
                elif key == "format_feedback":
                    # This should be a dict but ensure it's serializable
                    serialized_step[key] = self._serialize_dict(value)
                elif key == "state":
                    # Game state might contain complex objects - convert to string representation
                    serialized_step[key] = self._serialize_object(value)
                elif key == "step_info":
                    # Step info might contain complex objects - serialize safely
                    serialized_step[key] = self._serialize_dict(value)
                else:
                    # For any other fields, attempt safe serialization
                    serialized_step[key] = self._serialize_object(value)
            result["trajectory"].append(serialized_step)
        
        return result
    
    def _serialize_dict(self, obj: Any) -> Any:
        """Safely serialize a dictionary or dict-like object."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                result[str(key)] = self._serialize_object(value)
            return result
        return self._serialize_object(obj)
    
    def _serialize_object(self, obj: Any) -> Any:
        """Safely serialize any object to a JSON-compatible format."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_object(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._serialize_object(value) for key, value in obj.items()}
        else:
            # For any other object type, convert to string representation
            return str(obj)


def run_single_game(
    agents: Dict[str, ta.core.Agent],
    env_id: str,
    agent_order: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    format_requirement_level: str = "strict",
    track_tokens: bool = False
) -> GameInformation:
    """Run a single game with pre-created agents."""
    game_id = hashlib.md5(
        f"{env_id}_{'-'.join(agent_order)}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    # Setup
    env = ta.make(env_id)
    env.reset(num_players=len(agent_order),seed=metadata.get("replay_seed") if metadata else None) # seed is important for games with game inits like simplenegotiation

    player_agents = [agents[name] for name in agent_order]
    trajectory = []
    done = False
    
    # Track errors
    invalid_moves = {name: 0 for name in agent_order}
    format_errors = {name: 0 for name in agent_order}

    # Replay existing actions (if any)
    replay_actions = metadata.get("replay_actions") if metadata else None
    used_replay = False
    if replay_actions is not None:
        used_replay = True
        for action in replay_actions:
            # Figure out whose turn
            pid, obs = env.get_observation()
            agent_name = agent_order[pid]

            # We don’t call the agent, we already have the action
            raw_action = action[-1]
            parsed_action, format_feedback = extract_action_and_format_feedback(
                raw_action, env_id, format_requirement_level
            )

            # Step environment
            done, step_info = env.step(action=parsed_action)

            # Record step
            step_data = {
                "step": len(trajectory),
                "player_id": pid,
                "agent_name": agent_name,
                "observation": obs,
                "raw_action": raw_action,
                "action": parsed_action,
                "format_feedback": format_feedback,
                "state": copy.deepcopy(env.state.game_state),
                "step_info": step_info,
                "seed": env.state.seed,
                "replayed_action": True
            }
            trajectory.append(step_data)

            # Track invalid moves
            if step_info.get("invalid_move", False) or "invalid" in str(step_info.get("error", "")).lower():
                invalid_moves[agent_name] += 1

            # Track format errors
            if not format_feedback.get("correct_answer_format", False):
                format_errors[agent_name] += 1

            if done:
                break

        if not done:
            print("----------------playing from--------------------")
            print(env.get_observation())

    # Regardless, continue/start normal play
    while not done:
        pid, obs = env.get_observation()
        agent_name = agent_order[pid]

        # Get current game state before agent decision
        current_game_state = copy.deepcopy(env.state.game_state) if hasattr(env.state, 'game_state') else None
        
        # Get agent action with game state if agent supports it
        if hasattr(player_agents[pid], '__call__') and hasattr(player_agents[pid], 'use_state_abstracts_match'):
            # This is a MemoryEnhancedAgent that can use game state and will return observation with raw_action
            raw_action, obs = player_agents[pid](obs, game_state=current_game_state, player_id=pid)
        else:
            # Regular agent
            raw_action = player_agents[pid](obs, track_tokens=track_tokens)
        action, format_feedback = extract_action_and_format_feedback(
            raw_action, env_id, format_requirement_level
        )

        # Step environment
        done, step_info = env.step(action=action)

        # Record step
        step_data = {
            "step": len(trajectory),
            "player_id": pid,
            "agent_name": agent_name,
            "observation": obs,
            "raw_action": raw_action,
            "action": action,
            "format_feedback": format_feedback,
            "state": copy.deepcopy(env.state.game_state),
            "step_info": step_info,
            "seed": env.state.seed,
            "replayed_action": False
        }
        trajectory.append(step_data)

        # Track errors
        if step_info.get("invalid_move", False) or "invalid" in str(step_info.get("error", "")).lower():
            invalid_moves[agent_name] += 1
        if not format_feedback.get("correct_answer_format", False):
            format_errors[agent_name] += 1

    # --- Wrap up ---
    rewards, game_info = env.close()
    rewards_list = (
        [rewards.get(i, 0) for i in range(len(agent_order))]
        if isinstance(rewards, dict)
        else list(rewards)
    )
    winners = [i for i, r in enumerate(rewards_list) if r == 1]
    losers = [i for i, r in enumerate(rewards_list) if r == -1]

    return GameInformation(
        game_id=game_id,
        env_id=env_id,
        agent_names=agent_order,
        rewards=rewards_list,
        winners=winners,
        losers=losers if losers else None,
        trajectory=trajectory,
        game_info=game_info,
        timestamp=datetime.now().isoformat(),
        metadata=metadata or {},
        num_turns=len(trajectory),
        invalid_moves_per_agent=invalid_moves,
        format_errors_per_agent=format_errors,
        used_replay=used_replay
    )