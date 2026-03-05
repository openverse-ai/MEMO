"""
Agent Pool Manager - Manages tournament agents with template support and performance tracking
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
import textarena as ta
from dataclasses import dataclass, field
import trueskill
from mpr.cores.templates import apply_template, extract_action_and_format_feedback

from mpr.memory.trajectory_memory_system import TrajectoryMemorySystem

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """Track performance metrics for an agent."""
    agent_id: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_played: int = 0
    trueskill_rating: trueskill.Rating = field(default_factory=lambda: trueskill.Rating())
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def update_game_result(self, result: str):
        """Update game statistics."""
        self.games_played += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        elif result == "draw":
            self.draws += 1
    
    def get_fitness(self, method: str = "winrate") -> float:
        """Calculate fitness score.
        
        Args:
            method: "winrate" or "trueskill" (default: "winrate")
        """
        if method == "trueskill":
            # Use mu directly as the skill rating
            return self.trueskill_rating.mu
        else:  # Default to winrate
            return self.win_rate() * 100
        
    def rates_vs_opponents(self, perspective="self") -> Tuple[float, float, float]:
        """Get win, loss, draw rates from specified perspective."""
        if self.games_played == 0:
            return 0.0, 0.0, 0.0

        win = self.win_rate()
        draw = self.draws / self.games_played
        loss = 1 - win - draw

        if perspective == "baseline":
            return loss, win, draw

        return win, loss, draw


class TemplatedAgent(ta.agents.basic_agents.OpenRouterAgent):
    """Agent that applies templates to observations before processing."""

    def __init__(
        self,
        model_name: str,
        template_name: Optional[str] = None,
        evolved_prompt: Optional[str] = None,
        temperature: float = None,
        track_tokens: bool = True,
        contribute_to_selfplay: bool = False,
        contribute_to_optimization: bool = False,
        **kwargs
    ):
        """Initialize templated agent with temperature control and token tracking."""
        if temperature is not None:
            super().__init__(
                model_name=model_name,
                temperature=temperature,
                track_tokens=track_tokens,
                contribute_to_selfplay=contribute_to_selfplay,
                contribute_to_optimization=contribute_to_optimization,
                **kwargs
            )
        else:
            super().__init__(
                model_name=model_name,
                track_tokens=track_tokens,
                contribute_to_selfplay=contribute_to_selfplay,
                contribute_to_optimization=contribute_to_optimization,
                **kwargs
            )

        self.template_name = template_name
        self.evolved_prompt = evolved_prompt
        
    def __call__(self, observation: str, track_tokens: Optional[bool] = None) -> str:
        """Process observation with template."""
        # If we have an evolved prompt, use it with the template system
        if self.evolved_prompt:
            # Use the mpr-evolved template which properly formats the evolved prompt
            formatted_obs = apply_template("mpr-evolved", observation, self.evolved_prompt)
        elif self.template_name:
            # Use template without evolved prompt
            formatted_obs = apply_template(self.template_name, observation, None)
        else:
            # No template or evolved prompt
            formatted_obs = observation

        # Call parent with token-tracking control
        return super().__call__(formatted_obs, track_tokens=track_tokens)


class AgentPool:
    """Manages a pool of agents for tournaments with performance tracking."""
    
    def __init__(self, prompt_debug: bool = False):
        self.agents = {}
        self.model_names = []
        self.prompt_debug = prompt_debug
        self.templates = {}
        self.evolved_prompts = {}
        self.performance = {}  # Track agent performance
        self.trueskill_env = trueskill.TrueSkill(beta=4.0)
    
    def add_agent(
        self,
        agent_id: str,
        model_name: str,
        template_name: Optional[str] = None,
        evolved_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """Add an agent to the pool with template support."""
        agent = TemplatedAgent(
            model_name=model_name,
            template_name=template_name,
            evolved_prompt=evolved_prompt,
            temperature=temperature,
            contribute_to_selfplay=True
        )
        
        self.agents[agent_id] = agent
        self.model_names.append(model_name)
        self.templates[agent_id] = template_name
        if evolved_prompt:
            self.evolved_prompts[agent_id] = evolved_prompt
        
        # Initialize performance tracking
        self.performance[agent_id] = AgentPerformance(agent_id=agent_id)
        
        if self.prompt_debug:
            logger.info(f"Added agent {agent_id}: {model_name} with template {template_name}")
    
    def get_agent(self, agent_id: str) -> ta.core.Agent:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_all_agent_ids(self) -> List[str]:
        """Get all agent IDs."""
        return list(self.agents.keys())
    
    def get_agent_performance(self, agent_id: str) -> Optional[AgentPerformance]:
        """Get performance metrics for an agent."""
        return self.performance.get(agent_id)
    
    def update_game_result(self, agent_id: str, result: str, opponent_id: Optional[str] = None):
        """Update game result for an agent."""
        if agent_id in self.performance:
            self.performance[agent_id].update_game_result(result)
            logger.debug(f"Updated {agent_id}: {result}")
    
    def batch_update_trueskill(self, match_results: List[Dict]):
        """Batch update TrueSkill ratings after tournament completes.
        
        This avoids concurrency issues by updating all ratings at once after games finish.
        Automatically determines ranks from rewards.
        
        Args:
            match_results: List of match results, each containing:
                - 'agents': List of agent IDs that played
                - 'rewards': List of rewards (1=win, -1=loss, 0=draw)
                
        Example:
            match_results = [
                {'agents': ['agent1', 'agent2'], 'rewards': [1, -1]},  # agent1 beat agent2
                {'agents': ['agent3', 'agent4'], 'rewards': [0, 0]},  # draw
                {'agents': ['agent1', 'agent3', 'agent5'], 'rewards': [1, -1, -1]},  # agent1 wins
            ]
        """
        if self.prompt_debug:
            logger.info(f"Batch updating TrueSkill for {len(match_results)} matches")
        
        # Process each match result
        for match in match_results:
            agent_ids = match.get('agents', [])
            rewards = match.get('rewards', [])
            
            if len(agent_ids) != len(rewards):
                logger.warning(f"Skipping match: {len(agent_ids)} agents but {len(rewards)} rewards")
                continue
            
            # Convert rewards to ranks
            # Create list of (reward, index) pairs and sort by reward descending
            reward_indices = [(reward, i) for i, reward in enumerate(rewards)]
            reward_indices.sort(key=lambda x: x[0], reverse=True)
            
            # Assign ranks based on sorted rewards
            ranks = [0] * len(rewards)
            current_rank = 0
            prev_reward = None
            
            for reward, idx in reward_indices:
                if prev_reward is not None and reward < prev_reward:
                    current_rank += 1
                ranks[idx] = current_rank
                prev_reward = reward
            
            # Get current ratings
            current_ratings = []
            for agent_id in agent_ids:
                if agent_id not in self.performance:
                    # Initialize performance if not exists
                    self.performance[agent_id] = AgentPerformance(agent_id=agent_id)
                current_ratings.append(self.performance[agent_id].trueskill_rating)
            
            # Update ratings using TrueSkill
            # For individual players, wrap in tuples (required by trueskill library)
            rating_groups = [(rating,) for rating in current_ratings]
            
            try:
                # Calculate new ratings
                new_rating_groups = self.trueskill_env.rate(rating_groups, ranks=ranks)
                
                # Apply new ratings back
                for agent_id, new_rating_group in zip(agent_ids, new_rating_groups):
                    old_mu = self.performance[agent_id].trueskill_rating.mu
                    self.performance[agent_id].trueskill_rating = new_rating_group[0]
                    new_mu = new_rating_group[0].mu
                    logger.debug(f"Updated {agent_id} TrueSkill: {old_mu:.2f} -> {new_mu:.2f}")
                    
            except Exception as e:
                logger.warning(f"Failed to update TrueSkill for match: {e}")
        
        if self.prompt_debug:
            logger.info("TrueSkill batch update complete")
    
    def get_ranked_agents(self, fitness_method: str = "trueskill") -> List[Tuple[str, float]]:
        """Get agents ranked by fitness."""
        rankings = []
        for agent_id, perf in self.performance.items():
            fitness = perf.get_fitness(fitness_method)
            rankings.append((agent_id, fitness))
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def create_agents_from_models(
        self,
        model_names: List[str],
        templates: Optional[List[str]] = None,
        evolved_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ):
        """Create agents from model names with templates."""
        if templates and len(templates) != len(model_names):
            raise ValueError("Number of templates must match number of models")
        if evolved_prompts and len(evolved_prompts) != len(model_names):
            raise ValueError("Number of prompts must match number of models")
        
        for i, model_name in enumerate(model_names):
            agent_id = f"agent_{i}_{model_name.split('/')[-1]}"
            logger.info(f"DEBUG create_agents_from_models: Creating agent with ID '{agent_id}' for model '{model_name}' (index={i})")
            template = templates[i] if templates else None
            prompt = evolved_prompts[i] if evolved_prompts else None
            self.add_agent(agent_id, model_name, template, prompt, temperature)
    
    def create_prompt_agents_from_models(
        self,
        model_names: List[str],
        prompt_candidates: List[Any],  # List of PromptCandidate objects
        base_template: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """Create agents from prompt candidates with their evolved prompts and templates."""
        if len(prompt_candidates) != len(model_names):
            # If mismatch, cycle through candidates
            from itertools import cycle
            prompt_cycle = cycle(prompt_candidates)
            prompt_candidates = [next(prompt_cycle) for _ in model_names]
        
        for i, (model_name, candidate) in enumerate(zip(model_names, prompt_candidates)):
            # Always create new agent ID - no more preserved_agent_id logic
            agent_id = f"prompt_agent_{i}_{candidate.id}"
            
            # Add agent with evolved prompt
            self.add_agent(
                agent_id=agent_id,
                model_name=model_name,
                template_name="mpr-evolved",
                evolved_prompt=candidate.prompt,
                temperature=temperature
            )
            
            # # Carry over performance data if available
            # if hasattr(candidate, 'wins'):
            #     perf = self.performance[agent_id]
            #     perf.wins = candidate.wins
            #     perf.losses = candidate.losses
            #     perf.draws = candidate.draws
            #     perf.games_played = candidate.games_played
            #     if hasattr(candidate, 'trueskill_rating'):
            #         perf.trueskill_rating = candidate.trueskill_rating
            
            if self.prompt_debug:
                logger.info(f"Created prompt agent {agent_id} with evolved prompt from {candidate.id}")
    
    def update_prompt_for_agent(self, agent_id: str, new_prompt: str, new_template: Optional[str] = None):
        """Update the evolved prompt for an existing agent."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if isinstance(agent, TemplatedAgent):
                agent.evolved_prompt = new_prompt
                self.evolved_prompts[agent_id] = new_prompt
                
                if new_template:
                    agent.template_name = new_template
                    self.templates[agent_id] = new_template
                
                logger.info(f"Updated prompt for agent {agent_id}")
                return True
        return False
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all its associated data from the pool."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found in pool")
            return False
        
        # Remove from all tracking dictionaries
        if agent_id in self.agents:
            del self.agents[agent_id]
        
        if agent_id in self.performance:
            del self.performance[agent_id]
        
        if agent_id in self.templates:
            del self.templates[agent_id]
        
        if agent_id in self.evolved_prompts:
            del self.evolved_prompts[agent_id]
        
        # Remove from model_names list if present
        # Note: This is tricky as model_names doesn't track which agent uses which model
        # So we just log the deletion
        
        if self.prompt_debug:
            logger.info(f"Deleted agent {agent_id} from pool")
        return True
    
    def delete_agents(self, agent_ids: List[str]) -> int:
        """Delete multiple agents from the pool.
        
        Returns:
            Number of agents successfully deleted
        """
        deleted_count = 0
        for agent_id in agent_ids:
            if self.delete_agent(agent_id):
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} agents from pool")
        
        return deleted_count
    
    def clear_prompt_agents(self):
        """Remove all prompt agents (agents with 'prompt_agent_' prefix) from the pool."""
        prompt_agent_ids = [
            agent_id for agent_id in self.agents.keys() 
            if agent_id.startswith("prompt_agent_")
        ]
        
        deleted = self.delete_agents(prompt_agent_ids)
        logger.info(f"Cleared {deleted} prompt agents from pool")
        return deleted
    
    def create_memory_agents_from_models(
        self,
        model_names: List[str],
        memory_system: 'TrajectoryMemorySystem',
        memory_agents: Optional[List[int]] = None,
        templates: Optional[List[str]] = None,
        evolved_prompts: Optional[List[str]] = None
    ):
        """Create agents with memory enhancement for specified indices.
        
        Args:
            model_names: List of model names to create agents for
            memory_system: Pre-initialized memory system (REQUIRED)
            memory_agents: List of agent indices to enhance with memory (default: all)
            templates: List of templates for each model
            evolved_prompts: List of evolved prompts for each model
        """
        from mpr.memory.trajectory_memory_system import MemoryEnhancedAgent
        
        # Create basic agents first
        self.create_agents_from_models(model_names, templates, evolved_prompts)
        
        # Check if memory is available
        shared_memory = memory_system.get_shared_memory()
        if not shared_memory:
            logger.warning(f"No shared memory found. Agents will run without memory enhancement.")
            return
        else:
            logger.info(f"Loaded shared memory from generation {shared_memory.get('generation', 'unknown')}")
            logger.info(f"Memory based on {shared_memory.get('total_games', 0)} games with {shared_memory.get('performance', {}).get('overall_win_rate', 0):.1%} win rate")
        
        # Default to all agents having memory if not specified
        if memory_agents is None:
            memory_agents = list(range(len(model_names)))
        
        # Get agent IDs
        agent_ids = self.get_all_agent_ids()
        
        # Enhance selected agents with memory
        for i in memory_agents:
            if i < len(agent_ids):
                agent_id = agent_ids[i]
                base_agent = self.agents[agent_id]
                
                # Wrap with memory enhancement
                memory_agent = MemoryEnhancedAgent(
                    base_agent=base_agent,
                    agent_name=agent_id,
                    memory_system=memory_system
                )
                
                # Replace the original agent
                self.agents[agent_id] = memory_agent
                logger.info(f"Enhanced agent {agent_id} with memory")
    
    def reset(self):
        """Reset the agent pool."""
        self.agents.clear()
        self.model_names.clear()
        self.templates.clear()
        self.evolved_prompts.clear()