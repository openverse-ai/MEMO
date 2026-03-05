import math
import random
import hashlib
import json
from collections import deque
from typing import Any, Dict, Optional, List, Tuple

from mpr.cores.game_runner import GameInformation

class ReplayBuffer:
    def __init__(self, buffer_capacity: int = 1000, alpha: float = 0.6, beta: float = 0.4, max_steps: Optional[int] = None):
        self.buffer = deque(maxlen=buffer_capacity)
        self.alpha = alpha
        self.beta = beta
        self.sample_counts = {} 
        self.occurence_counts = {}
        self.max_steps = max_steps  # Maximum number of steps to store per game (None = all steps)

    def _hash_played_actions(self, played_actions: List[Tuple[int, str]]) -> str:
        """Compute the stable hash from a played sequence"""
        traj_str = repr(played_actions).encode("utf-8")
        return hashlib.sha256(traj_str).hexdigest()
    
    def push_batch(self, result: GameInformation) -> int:
        """Push all states from a trajectory. Returns number of episodes added."""
        trajectory = result.trajectory
        if not trajectory or len(trajectory) < 2:
            return 0
        
        if self.max_steps is not None:
            print(f"[ReplayBuffer] Limiting state storage to first {self.max_steps} steps per game")
        
        # Get the last non-terminal step observation and state
        last_step_idx = len(trajectory) - 2  # -1 is terminal, -2 is last playable step
        last_observation = trajectory[last_step_idx]["observation"]
        final_state = json.dumps(trajectory[last_step_idx]["state"])  # Final board state
 
        added = 0
        # Limit to max_steps if specified, otherwise process all non-terminal steps
        steps_to_process = trajectory[:-1]  # exclude terminal step
        if self.max_steps is not None:
            steps_to_process = steps_to_process[:self.max_steps]
        
        for idx, step_data in enumerate(steps_to_process):
            # Build differential by collecting future moves from this point
            future_moves = []
            for future_idx in range(idx + 1, len(trajectory) - 1):  # From next step to last playable
                future_step = trajectory[future_idx]
                future_player = future_step["player_id"]
                future_action = future_step["action"]
                future_moves.append(f"Player {future_player}: {future_action}")
            
            # If we have future moves, show them; otherwise indicate it's the last state
            if future_moves:
                full_game_actions = "\n".join(future_moves)
            else:
                full_game_actions = "This is the final state - no further moves"
            
            played_actions = [(t["player_id"], t["action"]) for t in trajectory[: idx + 1]]
            next_state = json.dumps(trajectory[idx + 1]["state"])
            
            # Get next action
            next_action = None
            if idx + 1 < len(trajectory) - 1:  # Check if there's a next non-terminal step
                next_action = trajectory[idx + 1]["action"]
            
            added += self.push(
                game_id=result.game_id,
                winners=result.winners,
                played_actions=played_actions,
                action=step_data["action"],
                state=json.dumps(step_data["state"]),
                next_state=next_state,
                metadata=result.metadata,
                reward=result.rewards,
                player_id=step_data["player_id"],
                seed=step_data["seed"],
                is_replayed_step=step_data["replayed_action"],
                full_game_actions=full_game_actions,  # Differential from current to last
                next_action=next_action,
                final_state=final_state,  # Add final game state
            )
        return added
    
    def push(
        self,
        game_id: str,
        winners: List[int],
        played_actions: List[Tuple[int, str]],
        action: str,
        state: Any,
        next_state: Any,
        metadata: Optional[Dict] = None,
        reward: Optional[float] = None,
        player_id: Optional[int] = None,
        seed: Optional[int] = None,
        is_replayed_step: Optional[bool] = None,
        full_game_actions: Optional[str] = None,
        next_action: Optional[str] = None,
        final_state: Optional[str] = None
    ):
        traj_hash = self._hash_played_actions(played_actions)

        # increment the occurence count for this trajectory
        # this is not affected by the popping of old entries from the buffer
        # it counts how many times we've seen this trajectory
        self.occurence_counts[traj_hash] = self.occurence_counts.get(traj_hash, 0) + 1
        inv_priority = 1.0 / self.occurence_counts[traj_hash]

        # if the buffer is full, we will remove the oldest entry
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()

        episode = {
            "game_id": game_id,
            "winners": winners,
            "metadata": metadata or {},
            "played_actions": list(played_actions),
            "action": action,
            "state": state,
            "next_state": next_state,
            "hash": traj_hash,
            "reward": reward if reward is not None else 0.0,
            "player_id": player_id,
            "priority": inv_priority,
            "seed": seed,
            "is_replayed_step": is_replayed_step,
            "full_game_actions": full_game_actions,  # Store complete game sequence
            "next_action": next_action,  # Store next action
            "final_state": final_state  # Store final game state
        }
        self.buffer.append(episode)

        # after pushing the states, we'd want to update the priority of 
        # all episodes in the buffer that
        for ep in self.buffer:
            if ep["hash"] == traj_hash:
                ep["priority"] = inv_priority

        return 1  # number of episodes added

    def sample_prioritized(self, batch_size: int = 1) -> List:
        """Sample using inverse-frequency priority"""
        if not self.buffer:
            return [] # if the buffer is not populated, then return it as empty list
        
        priorities = [ep.get("priority", 1e-6) ** self.alpha for ep in self.buffer]
        total = sum(priorities)
        probs = [p / total for p in priorities]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        batch = [list(self.buffer)[i] for i in indices]

        # for the sampled ones, increment the sample counts
        for ep in batch:
            h = ep["hash"]
            self.sample_counts[h] = self.sample_counts.get(h, 0) + 1
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear all buffer contents and reset statistics."""
        self.buffer.clear()
        self.sample_counts.clear()
        self.occurence_counts.clear()
    
    def sample_strategic_states(self, topk: int = 10) -> List[Tuple[str, Dict]]:
        """Find strategic states ranked by outcome variance (interestingness).

        Groups episodes by state, computes win/loss/draw stats, and returns
        the top-k states with the highest variance in outcomes.

        Args:
            topk: Maximum number of unique strategic states to return

        Returns:
            List of tuples: (state_json_str, stats_dict)
            stats_dict has keys: wins, losses, draws, count, player_id
        """
        if not self.buffer:
            return []

        # Group episodes by state
        state_stats = {}
        for ep in self.buffer:
            state_key = ep["state"]
            if state_key not in state_stats:
                state_stats[state_key] = {
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "count": 0,
                    "player_id": ep["player_id"],
                }
            state_stats[state_key]["count"] += 1

            if ep["player_id"] in ep["winners"]:
                state_stats[state_key]["wins"] += 1
            elif len(ep["winners"]) == 0:
                state_stats[state_key]["draws"] += 1
            else:
                state_stats[state_key]["losses"] += 1

        # Score by interestingness: states with mixed outcomes and sufficient depth
        scored = []
        for state_key, stats in state_stats.items():
            total = stats["count"]
            if total < 2:
                continue
            if stats["wins"] == 0:
                continue

            win_rate = stats["wins"] / total
            loss_rate = stats["losses"] / total
            interestingness = abs(win_rate - loss_rate) * math.log1p(total)
            scored.append((interestingness, state_key, stats))

        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for _, state_key, stats in scored[:topk]:
            result.append((state_key, stats))
        return result

    def debug_print(self, n: int = 5):
        print(f"🧾 ReplayBuffer contents (showing up to {n}/{len(self.buffer)} episodes)")
        for i, episode in enumerate(list(self.buffer)[-n:]):
            print(f"--- Episode {i+1} ---")
            print(f"Game ID: {episode['game_id']}")
            print(f"Winners: {episode['winners']}")
            print(f"Player: {episode.get('player_id', 'N/A')}")
            print(f"played_actions length: {len(episode['played_actions'])}")
            print(f"Reward: {episode.get('reward')}")
            print(f"Priority (inv-freq): {episode.get('priority', 0.0):.3f}")
            print(f"Occurrences: {self.occurence_counts.get(episode['hash'], 0)}")
            print(f"Game Seed: {str(episode['seed'])}")
            print(f"State (truncated): {str(episode['state'])[:200]}")
            print(f"Next State (truncated): {str(episode['next_state'])[:200]}")
            print(f"played_actions: {episode['played_actions']}")
            print(f"Is Replayed Step: {episode.get('is_replayed_step', 'N/A')}")
            print(f"Action taken: {episode.get('action', 'N/A')}")
            print(f"Next action: {episode.get('next_action', 'N/A')}")
            print(f"Final state (truncated): {str(episode.get('final_state', 'N/A'))[:200]}")
            print(f"Full game actions (truncated): {str(episode.get('full_game_actions', 'N/A'))[:200]}")
