"""
Prompt Evolution Engine for Self-Play Tournament System
Manages prompt evolution through tournaments with AgentPool integration.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict
import textarena as ta

if TYPE_CHECKING:
    from mpr.memory.trajectory_memory_system import TrajectoryMemorySystem

logger = logging.getLogger(__name__)

_STYLES = (
    "aggressive", "defensive", "analytical", "creative", "chain of thought",
    "strategic", "pattern-focused", "mathematical", "psychological", "balanced",
    "adaptive", "opportunistic", "conservative", "risk-taking", "methodical",
    "intuitive", "predictive", "reactive", "proactive", "experimental",
    "systematic", "positional", "territorial", "sacrificial", "blocking-focused",
    "center-control", "edge-control", "fork-creating", "trap-setting",
    "endgame-focused", "opening-focused", "calculating", "heuristic-based",
    "minimax-oriented", "probabilistic", "rule-based", "principle-driven",
    "context-aware", "meta-gaming", "counter-play", "exploitative",
    "deceptive", "transparent", "unpredictable", "consistent", "alternating",
    "escalating", "de-escalating", "mirroring", "contrarian", "harmonizing",
)


@dataclass
class PromptCandidate:
    """Represents a prompt candidate in evolution."""
    id: str
    prompt: str
    generation: int
    
    # Evolution metadata
    parent_id: Optional[str] = None
    creation_method: str = "unknown"  # "base", "variation", "elite", "random", "trajectory", "crossover"
    
    # Link to agent in pool
    agent_id: Optional[str] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "prompt": self.prompt,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "creation_method": self.creation_method,
            "agent_id": self.agent_id
        }


class PromptEvolutionEngine:
    """Manages prompt evolution through tournament-based selection."""
    
    def __init__(
        self,
        population_size: int = 10,
        analyzer_model: Optional[str] = None,
        env_id: str = "TicTacToe-v0",
        output_dir: str = "prompt_evolution",
        # Evolution strategy ratios (should sum to 1.0)
        keep_ratio: float = 0.3,  # Ratio of elites to keep
        random_ratio: float = 0.2,  # Ratio for pure random exploration (no memory)
        memory_guided_ratio: float = 0.0,  # Ratio for memory-guided generation using insights
        trajectory_ratio: float = 0.3,  # Ratio for trajectory-based improvements
        crossover_ratio: float = 0.2,  # Ratio for crossover (remainder)
        # Importance ranking feature
        use_importance_ranking: bool = False,  # Disable importance ranking by default for now
        # Memory system for random generation (always enabled)
        memory_system: Optional['TrajectoryMemorySystem'] = None,  # TrajectoryMemorySystem instance for memory access
        # Fitness method for ranking and selection
        fitness_method: str = "trueskill",  # "trueskill" or "winrate"
        # Temperature for model sampling
        temperature: float = 1.0,
        # Debug logging flag
        prompt_debug: bool = False,
        # Insight sampling mode for memory-guided generation
        insight_sampling_mode: str = "sample"  # "partition", "sample", or "single"
    ):
        self.population_size = population_size
        self.env_id = env_id
        self.generation = 0
        self.population: List[PromptCandidate] = []
        
        # Evolution ratios
        self.keep_ratio = keep_ratio
        self.random_ratio = random_ratio
        self.memory_guided_ratio = memory_guided_ratio
        self.trajectory_ratio = trajectory_ratio
        self.crossover_ratio = crossover_ratio
        
        # Importance ranking feature
        self.use_importance_ranking = use_importance_ranking

        # Memory system (always enabled for guided random generation)
        self.memory_system = memory_system
        
        # Fitness method for ranking and selection
        self.fitness_method = fitness_method
        
        # Temperature for model sampling
        self.temperature = temperature
        
        # Debug logging flag
        self.prompt_debug = prompt_debug
        
        # Insight sampling mode and tracking
        self.insight_sampling_mode = insight_sampling_mode
        self._insight_partitions = []  # For "partition" mode
        self._current_partition_idx = 0  # Track which partition to use next
        self._used_insight_indices = set()  # For "single" mode
        self._memory_guided_call_count = 0  # Track number of memory-guided prompt calls
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{output_dir}_{env_id}_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyzer for prompt generation (optional)
        self.analyzer = None
        if analyzer_model:
            self._setup_analyzer(analyzer_model)
    
    def _setup_analyzer(self, model_name: str):
        """Setup analyzer model for prompt generation."""
        try:
            # Try to use OpenAIOpenrouterAgent if available
            try:
                from textarena.agents import OpenAIOpenrouterAgent
                self.analyzer = OpenAIOpenrouterAgent(model_name=model_name)
                logger.info(f"Analyzer initialized with OpenAIOpenrouterAgent: {model_name}")
            except ImportError:
                # Fall back to OpenRouterAgent
                self.analyzer = ta.agents.OpenRouterAgent(model_name=model_name, contribute_to_optimization=True)
                logger.info(f"Analyzer initialized with OpenRouterAgent: {model_name}")
        except Exception as e:
            logger.error(f"Failed to setup analyzer: {e}")
            self.analyzer = None
    
    def create_initial_population(self, base_prompt: str) -> List[PromptCandidate]:
        """Create initial population with variations of base prompt.
        
        Generation 0: Create different style variations from base prompt
        """
        if self.prompt_debug:
            logger.info(f"Creating initial population of {self.population_size} candidates")
        population = []
        
        # Add base prompt
        base_candidate = PromptCandidate(
            id="gen0_base",
            prompt=base_prompt,
            generation=0,
            creation_method="base"
        )
        population.append(base_candidate)
        
        for i in range(self.population_size - 1):
            style = _STYLES[i % len(_STYLES)]
            variation = self._create_style_variation(base_prompt, style)
            
            candidate = PromptCandidate(
                id=f"gen0_{style}_{i}",
                prompt=variation,
                generation=0,
                creation_method="variation",
                parent_id="gen0_base"
            )
            population.append(candidate)
        
        self.population = population
        self.generation = 0
        logger.info(f"Created {len(population)} initial candidates")
        return population
    
    def _create_style_variation(self, base_prompt: str, style: str) -> str:
        """Create a style variation of the base prompt."""
        if not self.analyzer:
            raise ValueError(f"Analyzer not initialized but required for style variation")
            
        prompt = f"""Transform this {self.env_id} prompt into a {style} style:

Original: {base_prompt}

Create a {style} version that:
1. Keeps the core strategy
2. Emphasizes {style} gameplay
3. Stays concise (1-3 sentences)

{style.capitalize()} version:"""
        
        
        response = self.analyzer(prompt)
        variation = response.strip().strip('"')
        if self.prompt_debug:
            logger.info(f"Created {style} variation: {variation[:100]}...")
        return variation

            
    def setup_agents_in_pool(self, agent_pool, model_names: List[str]):
        """Create prompt agents in AgentPool for current population."""
        if self.prompt_debug:
            logger.info(f"Setting up {len(self.population)} agents in pool")
        
        # Create agents using the pool's method
        agent_pool.create_prompt_agents_from_models(
            model_names=model_names,
            prompt_candidates=self.population,
            temperature=self.temperature
        )        
        # Update agent IDs in candidates
        for candidate in self.population:
            # Find the newly created agent for this candidate
            for agent_id in agent_pool.get_all_agent_ids():
                if candidate.id in agent_id:
                    candidate.agent_id = agent_id
                    break
  
    def _reset_insight_tracking(self, memory_guided_count: int):
        """Reset insight tracking for a new generation of memory-guided prompts."""
        self._memory_guided_call_count = 0
        self._used_insight_indices.clear()
        self._current_partition_idx = 0
        self._insight_partitions = []

        # Prepare partitions if in partition mode
        if self.insight_sampling_mode == "partition":
            shared_memory = self.memory_system.get_shared_memory()
            if shared_memory and 'insights' in shared_memory:
                insights = shared_memory['insights']
                if insights and memory_guided_count > 0:
                    # Partition insights into memory_guided_count groups
                    partition_size = max(1, len(insights) // memory_guided_count)
                    for i in range(memory_guided_count):
                        start_idx = i * partition_size
                        if i == memory_guided_count - 1:
                            # Last partition gets all remaining insights
                            self._insight_partitions.append(list(range(start_idx, len(insights))))
                        else:
                            end_idx = start_idx + partition_size
                            self._insight_partitions.append(list(range(start_idx, end_idx)))

                    if self.prompt_debug:
                        logger.info(f"Created {len(self._insight_partitions)} insight partitions for {memory_guided_count} memory-guided prompts")
                        for i, partition in enumerate(self._insight_partitions):
                            logger.info(f"  Partition {i}: {len(partition)} insights (indices {min(partition)}-{max(partition)})")
    
    def _verify_and_complete_population(self, new_population: List[PromptCandidate],
                                         elite_count: int, random_count: int, memory_guided_count: int,
                                         trajectory_count: int, crossover_count: int) -> List[PromptCandidate]:
        """Verify population size is correct and fill/trim as needed.

        Args:
            new_population: Current population list
            elite_count: Number of elite candidates added
            random_count: Number of random candidates added
            memory_guided_count: Number of memory-guided candidates added
            trajectory_count: Number of trajectory candidates added
            crossover_count: Number of crossover candidates added

        Returns:
            Verified and corrected population list
        """
        if len(new_population) == self.population_size:
            return new_population

        actual_total = elite_count + random_count + memory_guided_count + trajectory_count + crossover_count
        logger.error(f"Population size mismatch: expected {self.population_size}, got {len(new_population)}")
        logger.error(f"Breakdown: {elite_count} elites + {random_count} random + {memory_guided_count} memory_guided + {trajectory_count} trajectory + {crossover_count} crossover = {actual_total}")
        logger.error(f"Ratios: keep={self.keep_ratio}, random={self.random_ratio}, memory_guided={self.memory_guided_ratio}, trajectory={self.trajectory_ratio}, crossover={self.crossover_ratio}, sum={self.keep_ratio + self.random_ratio + self.memory_guided_ratio + self.trajectory_ratio + self.crossover_ratio}")
        
        if len(new_population) < self.population_size:
            # Fill remaining with random generation
            shortage = self.population_size - len(new_population)
            logger.warning(f"Ratios sum to {self.keep_ratio + self.random_ratio + self.memory_guided_ratio + self.trajectory_ratio + self.crossover_ratio:.2f}, filling {shortage} remaining slots with random generation")
            
            for i in range(shortage):
                random_prompt = self._generate_random_prompt()
                fill_candidate = PromptCandidate(
                    id=f"gen{self.generation + 1}_fill{i}",
                    prompt=random_prompt,
                    generation=self.generation + 1,
                    creation_method="random_fill",
                    parent_id=None
                )
                new_population.append(fill_candidate)
            
            logger.info(f"Filled {shortage} slots with random generation")
        elif len(new_population) > self.population_size:
            # Trim excess
            new_population = new_population[:self.population_size]
            logger.warning(f"Trimmed population to {self.population_size}")
        
        return new_population
    
    def evolve_generation(self, agent_pool, base_prompt: str = None, trajectory_path: str = None) -> List[PromptCandidate]:
        """Evolve to next generation based on tournament results.
        
        Steps:
        1. Select top N elites
        2. Generate new random prompts
        3. Update prompts from trajectory feedback
        4. Create crossover prompts
        5. Remove old agents and add new ones to pool
        """
        logger.info(f"Evolving to generation {self.generation + 1}")
        
        # Get performance rankings from AgentPool
        rankings = agent_pool.get_ranked_agents(fitness_method=self.fitness_method)
        
        # Map agent IDs back to candidates
        candidate_performance = {}
        for candidate in self.population:
            if candidate.agent_id:
                for agent_id, trueskill_mu in rankings:
                    if agent_id == candidate.agent_id:
                        candidate_performance[candidate.id] = trueskill_mu
                        break
        
        # Sort candidates by TrueSkill rating
        sorted_candidates = sorted(
            self.population,
            key=lambda c: candidate_performance[c.id] if c.id in candidate_performance else 0,
            reverse=True
        )
        
        new_population = []
        
        # 1. Keep top elites based on keep_ratio
        elite_count = int(self.population_size * self.keep_ratio)
        elite_count = min(elite_count, len(sorted_candidates))
        for i in range(elite_count):
            elite = sorted_candidates[i]
            new_elite = PromptCandidate(
                id=f"gen{self.generation + 1}_elite{i}",
                prompt=elite.prompt,
                generation=self.generation + 1,
                creation_method="elite",
                parent_id=elite.id
            )
            # Don't preserve agent_id to avoid performance accumulation
            # We preserve the prompt but create fresh agents each generation
            # new_elite.preserved_agent_id = None  # Explicitly don't preserve
            new_population.append(new_elite)
        
        logger.info(f"Kept {elite_count} elites")
        
        # 2. Generate new random prompts (based on random_ratio) - pure random, no memory
        random_count = int(self.population_size * self.random_ratio)

        if random_count > 0:
            for i in range(random_count):
                random_prompt = self._generate_random_prompt()
                random_candidate = PromptCandidate(
                    id=f"gen{self.generation + 1}_random{i}",
                    prompt=random_prompt,
                    generation=self.generation + 1,
                    creation_method="random",
                    parent_id=None
                )
                new_population.append(random_candidate)

        logger.info(f"Generated {random_count} random prompts")

        # 2.5. Generate memory-guided prompts (based on memory_guided_ratio)
        memory_guided_count = int(self.population_size * self.memory_guided_ratio)

        # Reset insight tracking for memory-guided generation
        self._reset_insight_tracking(memory_guided_count)

        if memory_guided_count > 0:
            for i in range(memory_guided_count):
                memory_guided_prompt = self._generate_memory_guided_prompt()
                memory_guided_candidate = PromptCandidate(
                    id=f"gen{self.generation + 1}_memory_guided{i}",
                    prompt=memory_guided_prompt,
                    generation=self.generation + 1,
                    creation_method="memory_guided",
                    parent_id=None
                )
                new_population.append(memory_guided_candidate)

        logger.info(f"Generated {memory_guided_count} memory-guided prompts")
        
        # 3. Update from trajectory feedback (based on trajectory_ratio)
        trajectory_count = int(self.population_size * self.trajectory_ratio)
        for i in range(trajectory_count):
            if i < len(sorted_candidates):
                parent = sorted_candidates[i]
                updated_prompt = self._update_from_trajectory(parent, agent_pool, trajectory_path)
                trajectory_candidate = PromptCandidate(
                    id=f"gen{self.generation + 1}_trajectory{i}",
                    prompt=updated_prompt,
                    generation=self.generation + 1,
                    creation_method="trajectory",
                    parent_id=parent.id
                )
                new_population.append(trajectory_candidate)
        
        logger.info(f"Generated {trajectory_count} trajectory-based updates")
        
        # 4. Generate crossover candidates (based on crossover_ratio)
        crossover_count = int(self.population_size * self.crossover_ratio)
        
        if crossover_count > 0 and len(sorted_candidates) >= 2:
            for i in range(crossover_count):
                if len(new_population) >= self.population_size:
                    break
                    
                # Select two parents from top half
                parent1 = random.choice(sorted_candidates[:len(sorted_candidates)//2])
                parent2 = random.choice(sorted_candidates[:len(sorted_candidates)//2])
                while parent2.id == parent1.id and len(sorted_candidates) > 1:
                    parent2 = random.choice(sorted_candidates[:len(sorted_candidates)//2])
                
                crossover_prompt = self._crossover(parent1, parent2)
                crossover_candidate = PromptCandidate(
                    id=f"gen{self.generation + 1}_cross{i}",
                    prompt=crossover_prompt,
                    generation=self.generation + 1,
                    creation_method="crossover",
                    parent_id=f"{parent1.id}+{parent2.id}"
                )
                new_population.append(crossover_candidate)
        
        logger.info(f"Generated {crossover_count} crossover prompts")
        
        # Verify population is complete
        new_population = self._verify_and_complete_population(
            new_population, elite_count, random_count, memory_guided_count, trajectory_count, crossover_count
        )
        
        # Update generation
        self.generation += 1
        self.population = new_population
        
        logger.info(f"Evolution complete: Generation {self.generation} with {len(new_population)} candidates")
        return new_population
    
    def _generate_random_prompt(self) -> str:
        """Generate a new purely random prompt without using memory insights.

        This creates novel prompts through random exploration without any
        guidance from past game insights.
        """
        if not self.analyzer:
            raise ValueError("Analyzer required for random prompt generation")

        prompt = f"""Generate a creative and unique strategy prompt for {self.env_id}.

Requirements:
1. Be innovative and different from standard approaches
2. Focus on winning strategy
3. Keep it concise (1-3 sentences)

Creative prompt:"""

        response = self.analyzer(prompt)
        return response.strip().strip('"')

    def _generate_memory_guided_prompt(self) -> str:
        """Generate a new prompt using memory bank insights.

        Uses the configured insight sampling mode to select insights from
        shared memory and generate prompts guided by past game experiences.
        """
        if not self.analyzer:
            raise ValueError("Analyzer required for memory-guided prompt generation")

        # Try to get shared memory for memory-guided generation
        shared_memory = self.memory_system.get_shared_memory() if self.memory_system else None

        # Use memory-guided generation if shared_memory exists
        if shared_memory:
            insights = shared_memory['insights']
            if not insights:
                logger.warning("No insights available in shared memory, falling back to random generation")
                return self._generate_random_prompt()

            # Select insights based on sampling mode
            selected_insights = []

            if self.insight_sampling_mode == "partition":
                # Mode 1: Use non-overlapping partitions
                if self._current_partition_idx < len(self._insight_partitions):
                    partition_indices = self._insight_partitions[self._current_partition_idx]
                    selected_insights = [insights[idx] for idx in partition_indices if idx < len(insights)]
                    self._current_partition_idx += 1

                    if self.prompt_debug:
                        logger.info(f"Partition mode: Using partition {self._current_partition_idx-1} with {len(selected_insights)} insights")
                else:
                    logger.warning(f"No more partitions available (used {self._current_partition_idx}/{len(self._insight_partitions)})")
                    selected_insights = insights[:8]  # Fallback

            elif self.insight_sampling_mode == "sample":
                # Mode 2: Random sample with possible overlap
                sample_size = min(8, len(insights))  # Use up to 8 insights per prompt
                selected_indices = random.sample(range(len(insights)), sample_size)
                selected_insights = [insights[idx] for idx in selected_indices]

                if self.prompt_debug:
                    logger.info(f"Sample mode: Randomly selected {len(selected_insights)} insights from {len(insights)} total")

            elif self.insight_sampling_mode == "single":
                # Mode 3: Use one insight per call
                available_indices = set(range(len(insights))) - self._used_insight_indices

                if available_indices:
                    selected_idx = random.choice(list(available_indices))
                    selected_insights = [insights[selected_idx]]
                    self._used_insight_indices.add(selected_idx)

                    if self.prompt_debug:
                        logger.info(f"Single mode: Using insight {selected_idx} ({len(self._used_insight_indices)}/{len(insights)} used)")
                else:
                    # All insights have been used, reset and start over
                    logger.info("Single mode: All insights used, resetting and starting over")
                    self._used_insight_indices.clear()
                    selected_idx = random.choice(range(len(insights)))
                    selected_insights = [insights[selected_idx]]
                    self._used_insight_indices.add(selected_idx)

            else:
                # Invalid mode, fallback to sample mode
                logger.warning(f"Invalid insight_sampling_mode: {self.insight_sampling_mode}, using sample mode")
                sample_size = min(8, len(insights))
                selected_indices = random.sample(range(len(insights)), sample_size)
                selected_insights = [insights[idx] for idx in selected_indices]

            # Build memory context from selected insights
            memory_context = "\nKEY INSIGHTS FROM SHARED MEMORY:\n"
            for i, insight in enumerate(selected_insights, 1):
                memory_context += f"{i}. {insight}\n"

            # Increment call count
            self._memory_guided_call_count += 1

            # Generate prompt using memory context
            generation_prompt = f"""Generate a creative and strategic prompt for {self.env_id} using insights from shared memory.

{memory_context}

Requirements:
1. Incorporate the key insights and successful patterns from memory
2. Be innovative but grounded in proven approaches
3. Focus on winning strategy
4. Keep it concise (1-3 sentences)
5. Ensure proper move formatting to avoid format errors

Creative memory-guided prompt:"""

            response = self.analyzer(generation_prompt)
            generated_prompt = response.strip().strip('"')

            if self.prompt_debug:
                logger.info(f"Generated memory-guided prompt #{self._memory_guided_call_count} using {len(selected_insights)} insights")

            return generated_prompt

        # Fallback to random generation if no memory available yet
        logger.warning("No shared memory available for memory-guided generation, falling back to random")
        return self._generate_random_prompt()
    
    def _reflect_on_trajectory(self, game_data: dict, candidate_prompt: str, game_result: str) -> dict:
        """Generate reflection for a single game trajectory with full context."""
        our_moves = game_data['our_moves']
        full_trajectory = game_data.get('full_trajectory', [])
        format_errors = game_data['format_errors']
        total_moves = game_data['total_moves']
        
        # Build a summary of the game flow with states and moves
        game_flow = []
        if full_trajectory:
            # Show first few moves with state context
            for i, step in enumerate(full_trajectory):  
                if step['is_our_move']:
                    game_flow.append(f"  Our move: {step['action']}")
                else:
                    game_flow.append(f"  Opponent: {step['action']}")
                    
                # Add state snapshot every few moves
                if i % 3 == 0:
                    if 'state' in step:
                        game_flow.append(f"  [State after move {i+1}: {step['state']}...]")
        
        game_flow_str = '\n'.join(game_flow) if game_flow else 'No moves available'
        
        # Format errors are HIGH PRIORITY
        format_error_note = f"\nFORMAT ERRORS: {format_errors}/{total_moves} moves had incorrect format - THIS IS CRITICAL!, the final answer should be enclose by \\boxed{{}}\n highlight that in your feedback" if format_errors > 0 else ""
        
        # Get game rules from game_data (already extracted during trajectory processing)
        game_rules = game_data.get('game_rules',[])
        if not game_rules:
            game_rules = "No game rules available"
        
        reflection_prompt = f"""Analyze this {self.env_id} game trajectory with full context:

GAME RULES AND CONTEXT:
{game_rules}

GAME RESULT: {game_result.upper()}
OPPONENT: {game_data['opponent_name']}
GAME FLOW:
{game_flow_str}


Provide analysis in this exact format:

REFLECTION: [2-3 sentences analyzing the game progression, how our moves responded to opponent moves and game state, what strategy was followed, and what could be improved{'. MUST address the format errors as highest priority!' if format_errors > 0 else ''}]

ACTIONABLE_INSIGHT: [One specific, concrete improvement based on the game flow and state transitions]"""


        response = self.analyzer(reflection_prompt)
        
        # Parse the response
        reflection_text = ""
        insight = ""
        
        for line in response.strip().split('\n'):
            if line.startswith('REFLECTION:'):
                reflection_text = line.replace('REFLECTION:', '').strip()
            elif line.startswith('ACTIONABLE_INSIGHT:'):
                insight = line.replace('ACTIONABLE_INSIGHT:', '').strip()
        
        result = {
            'reflection': reflection_text if reflection_text else f"Game ended in {game_result} after {total_moves} moves.",
            'insight': insight,
            'format_errors': format_errors,
            'total_moves': total_moves,
            'reflection_prompt': reflection_prompt  # Save the prompt for memory
        }
        
        # Don't print reflections, just return
        return result
    
    def _reflect_on_trajectory_memory(self, game_data: dict, candidate_prompt: str, game_result: str) -> dict:
        """Generate reflection for a single game trajectory - simplified version."""
        last_observation = game_data['last_observation']
        last_action = game_data['last_action']
        our_moves = game_data['our_moves']
        format_errors = game_data['format_errors']
        total_moves = game_data['total_moves']
        
        # Simplified game summary
        player_idx = game_data['agent_idx']
        game_summary = f"\n{last_observation}\n\nPlayer {player_idx} action: {last_action}"
        
        # Format errors are HIGH PRIORITY
        format_error_note = f"\nFORMAT ERRORS: {format_errors}/{total_moves} moves had incorrect format - THIS IS CRITICAL!, the final answer should be enclose by \\boxed{{}}\n highlight that in your feedback" if format_errors > 0 else ""
        
        # Use outcome reason from game_info if available, otherwise fall back to basic description
        outcome_reason = game_data['outcome_reason']

        # Build insight instruction based on merge style
        use_fewshot = hasattr(self.memory_system, 'memory_merge_style') and "fewshot" in self.memory_system.memory_merge_style
        insight_instruction = (
            '[One specific, concrete improvement based on the game outcome, followed by one or two specific examples showing the game state and corresponding action. Format: "Improvement description. Example 1: When [game state], take action [specific action]. Example 2 (if applicable): When [different game state], take action [specific action]."]'
            if use_fewshot else
            "[One specific, concrete improvement based on the game outcome]"
        )

        format_priority = ". MUST address the format errors as highest priority!" if format_errors > 0 else ""

        reflection_prompt = f"""Analyze this {self.env_id} game outcome:

GAME OUTCOME: {outcome_reason}
TOTAL MOVES: {total_moves}
{format_error_note}

{game_summary}

Provide analysis in this exact format:

REFLECTION: [2-3 sentences analyzing the final game state, what strategy appeared to work or fail, and what could be improved{format_priority}]

ACTIONABLE_INSIGHT: {insight_instruction}"""


        # Try to get response with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.analyzer(reflection_prompt)
                
                # Parse the response
                reflection_text = ""
                insight = ""
                
                for line in response.strip().split('\n'):
                    if line.startswith('REFLECTION:'):
                        reflection_text = line.replace('REFLECTION:', '').strip()
                    elif line.startswith('ACTIONABLE_INSIGHT:'):
                        insight = line.replace('ACTIONABLE_INSIGHT:', '').strip()
                
                # Check if we got valid responses
                if reflection_text and insight:
                    break  # Success, exit retry loop
                
                # If missing required fields, retry
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}: Missing REFLECTION or ACTIONABLE_INSIGHT in response. Retrying...")
                    # Add explicit format reminder for retry
                    reflection_prompt = f"""

Provide analysis in this exact format:

REFLECTION: [2-3 sentences analyzing the final game state, what strategy appeared to work or fail, and what could be improved{'. MUST address the format errors as highest priority!' if format_errors > 0 else ''}]

ACTIONABLE_INSIGHT: [One specific, concrete improvement based on the game outcome]
{response}
"""
                
            except Exception as e:
                logger.error(f"Error in reflection analysis attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Final attempt failed, use defaults
                    reflection_text = f"Game ended in {game_result} after {total_moves} moves."
                    insight = "Unable to generate insight due to analysis error."
        
        # Use defaults if still missing after retries
        result = {
            'reflection': reflection_text if reflection_text else f"Game ended in {game_result} after {total_moves} moves.",
            'insight': insight if insight else "Focus on improving game strategy.",
            'format_errors': format_errors,
            'total_moves': total_moves,
            'reflection_prompt': reflection_prompt  # Save the prompt for memory
        }
        
        # Don't print reflections, just return
        return result
    
    def _synthesize_reflections(self, reflections: list, overall_stats: dict, candidate_prompt: str) -> tuple:
        """Synthesize reflections with optional importance ranking."""
        
        if self.use_importance_ranking:
            # Step 1: Evaluate importance across all trajectories
            reflections_for_importance = []
            for r in reflections:
                reflection_summary = f"[{r['result'].upper()}] {r['reflection']}"
                if r['format_errors'] > 0:
                    reflection_summary += f" (FORMAT ERRORS: {r['format_errors']}/{r['total_moves']})"
                reflections_for_importance.append(reflection_summary)
            
            importance_prompt = f"""You need to evaluate the relative importance of these game trajectories for improving the prompt.

CURRENT PROMPT: {candidate_prompt}

OVERALL PERFORMANCE:
- Win Rate: {overall_stats['win_rate']:.1%}
- Record: {overall_stats['wins']}W / {overall_stats['losses']}L / {overall_stats['draws']}D

GAME TRAJECTORIES:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(reflections_for_importance))}

For each trajectory, assign an importance score from 1-5:
- 5: Critical finding (e.g., consistent failure pattern, format errors, major strategic flaw)
- 4: Important pattern (e.g., recurring weakness, significant tactical issue)
- 3: Useful insight (e.g., moderate improvement opportunity)
- 2: Minor observation (e.g., small refinement possible)
- 1: Low relevance (e.g., isolated incident, already performing well)

Respond with ONLY a JSON array of importance scores in order, e.g., [3, 5, 2, 4, ...]"""

            importance_scores = []
            try:
                response = self.analyzer(importance_prompt)
                # Extract JSON array from response
                import json
                import re
                
                # Find JSON array in response
                json_match = re.search(r'\[[\d,\s]+\]', response)
                if json_match:
                    importance_scores = json.loads(json_match.group())
                    # Ensure we have the right number of scores
                    if len(importance_scores) != len(reflections):
                        logger.warning(f"Importance scores count mismatch: got {len(importance_scores)}, expected {len(reflections)}")
                        # Pad or truncate as needed
                        while len(importance_scores) < len(reflections):
                            importance_scores.append(3)  # Default importance
                        importance_scores = importance_scores[:len(reflections)]
                else:
                    logger.warning("Could not extract importance scores, using defaults")
                    importance_scores = [3] * len(reflections)
            except Exception as e:
                logger.warning(f"Failed to evaluate importance: {e}")
                importance_scores = [3] * len(reflections)
            
            # Step 2: Add importance scores to reflections
            for i, r in enumerate(reflections):
                r['importance'] = importance_scores[i] if i < len(importance_scores) else 3
            
            # Step 3: Sort reflections by importance
            reflections_sorted = sorted(reflections, key=lambda x: x['importance'], reverse=True)
        else:
            # Skip importance ranking - just use default importance of 3 and original order
            for r in reflections:
                r['importance'] = 3  # Default importance for all reflections
            reflections_sorted = reflections  # Keep original order
        
        # Step 4: Synthesize with importance weighting
        reflections_text = ""
        insights_text = ""
        
        # Include top reflections by importance (show more if we have more data)
        max_to_show = min(45, len(reflections_sorted))  # Show up to 15 most important
        for i, r in enumerate(reflections_sorted[:max_to_show]):
            importance = r['importance']
            importance_marker = f"[{importance}/5]"
            reflections_text += f"\nGame ({r['result']}) {importance_marker}: {r['reflection']}"
            if r['format_errors'] > 0:
                reflections_text += f" (FORMAT ERRORS: {r['format_errors']}/{r['total_moves']})"
            if r['insight']:
                insights_text += f"\n- {r['insight']} (importance: {importance})"
        
        # Create synthesis prompt based on whether importance ranking is used
        if self.use_importance_ranking:
            reflections_header = "GAME REFLECTIONS (sorted by importance, 1 to 5):"
            synthesis_task = """SYNTHESIS TASK:
Focus on the highest importance reflections (4-5 stars) as they indicate critical patterns.
Identify the most consistent issues and successful strategies across games.
Synthesize into 3-4 concrete strategic improvements."""
        else:
            reflections_header = "GAME REFLECTIONS:"
            synthesis_task = """SYNTHESIS TASK:
Identify the most consistent issues and successful strategies across games.
Synthesize into 3-4 concrete strategic improvements."""

        synthesis_prompt = f"""Synthesize these game analyses into a strategic improvement plan:

CURRENT PROMPT: {candidate_prompt}

OVERALL PERFORMANCE:
- Win Rate: {overall_stats['win_rate']:.1%}
- Record: {overall_stats['wins']}W / {overall_stats['losses']}L / {overall_stats['draws']}D

{reflections_header}
{reflections_text}

KEY INSIGHTS FROM GAMES:
{insights_text if insights_text else "No specific insights extracted"}

{synthesis_task}

STRATEGIC SYNTHESIS:"""

        try:
            synthesis = self.analyzer(synthesis_prompt)
            # Return both synthesis and updated reflections with importance
            return synthesis.strip(), reflections
        except Exception as e:
            logger.warning(f"Failed to synthesize reflections: {e}")
            return "Unable to synthesize reflections. Focus on balanced strategic play.", reflections
    
    def _load_trajectory_files(self, trajectory_path: str) -> List[Path]:
        """Load trajectory files from path (supports globs, directories, semicolons)."""
        import glob as glob_module

        trajectory_files = []
        for path in trajectory_path.split(';'):
            path = path.strip()
            if not path:
                continue
            if '*' in path:
                trajectory_files.extend([Path(f) for f in glob_module.glob(path)])
            else:
                p = Path(path)
                if p.exists():
                    if p.is_dir():
                        trajectory_files.extend(list(p.glob("*.json")))
                    elif p.is_file() and p.suffix == '.json':
                        trajectory_files.append(p)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for f in trajectory_files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    def _extract_game_data_from_trajectories(self, trajectory_files: List[Path], candidate: PromptCandidate) -> List[Dict]:
        """Extract game data for a candidate from trajectory files."""
        game_data_list = []

        for file in trajectory_files:
            with open(file, 'r') as f:
                data = json.load(f)

            for game in data if isinstance(data, list) else [data]:
                if candidate.agent_id not in str(game['agent_names']):
                    continue

                agent_idx = None
                for i, name in enumerate(game['agent_names']):
                    if candidate.agent_id in name:
                        agent_idx = i
                        break

                if agent_idx is None or 'rewards' not in game:
                    continue

                reward = game['rewards'][agent_idx]
                game_result = 'win' if reward > 0 else 'loss' if reward < 0 else 'draw'

                full_trajectory = []
                our_moves = []
                format_errors = 0
                game_rules = ""

                for step in game.get('trajectory', []):
                    step_data = {
                        "step": step['step'], "player_id": step['player_id'],
                        "action": step['action'], "agent_name": step['agent_name'],
                        "is_our_move": candidate.agent_id in step['agent_name'],
                        "observation": step['observation'], "format_feedback": step['format_feedback']
                    }
                    full_trajectory.append(step_data)

                    if not game_rules and step_data['is_our_move'] and step_data['observation']:
                        obs = step_data['observation']
                        if '[GAME] Current Board:' in obs:
                            game_rules = obs.split('[GAME] Current Board:')[0].strip()
                        elif 'Current Board:' in obs:
                            game_rules = obs.split('Current Board:')[0].strip()
                        else:
                            game_rules = obs[:1000] if len(obs) > 1000 else obs

                    if step_data['is_our_move']:
                        our_moves.append(step_data['action'])
                        if not step_data['format_feedback']['correct_answer_format']:
                            format_errors += 1

                opponent_idx = 1 - agent_idx if len(game['agent_names']) == 2 else None
                opponent_name = game['agent_names'][opponent_idx] if opponent_idx is not None else 'unknown'

                game_data_list.append({
                    'full_trajectory': full_trajectory, 'our_moves': our_moves,
                    'our_agent_idx': agent_idx, 'opponent_name': opponent_name,
                    'result': game_result, 'format_errors': format_errors,
                    'total_moves': len(our_moves), 'total_game_moves': len(full_trajectory),
                    'initial_state': full_trajectory[0].get('state', '') if full_trajectory else '',
                    'final_state': full_trajectory[-1].get('state', '') if full_trajectory else '',
                    'game_rules': game_rules
                })

        return game_data_list

    def _parse_json_response(self, response: str, max_retries: int = 5) -> dict:
        """Parse JSON with 'thinking' and 'final_prompt' fields, retrying via analyzer on failure."""
        _JSON_TEMPLATE = '{\n  "thinking": "Your complete analysis here.",\n  "final_prompt": "The improved prompt."\n}'

        current_response = response
        for attempt in range(max_retries + 1):
            json_start = current_response.find('{')
            json_end = current_response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                try:
                    result = json.loads(current_response[json_start:json_end])
                    if 'thinking' in result and 'final_prompt' in result:
                        return result
                    raise ValueError(f"Missing fields: got {list(result.keys())}")
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt >= max_retries:
                        raise ValueError(f"Failed to parse JSON after {max_retries + 1} attempts: {e}")
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
            else:
                if attempt >= max_retries:
                    raise ValueError(f"No valid JSON found after {max_retries + 1} attempts")
                logger.warning(f"No JSON structure found on attempt {attempt + 1}")

            # Retry with format correction prompt
            retry_prompt = f"""Your previous response was:\n\n{current_response}\n\nPlease provide your response as valid JSON with exactly these fields:\n\n{_JSON_TEMPLATE}\n\nRespond with ONLY the JSON object."""
            current_response = self.analyzer(retry_prompt)
            logger.info(f"Retry attempt {attempt + 1} for JSON parsing")

        raise ValueError(f"Failed to get valid JSON after {max_retries + 1} attempts")

    def _update_from_trajectory(self, candidate: PromptCandidate, agent_pool, trajectory_path: str = None) -> str:
        """Update prompt based on trajectory feedback from actual games.

        Pipeline: load trajectories → extract game data → reflect → synthesize → improve prompt.
        """
        if not self.analyzer:
            raise ValueError("Analyzer required for trajectory-based improvement")
        if not trajectory_path:
            raise ValueError("Trajectory path is required for trajectory-based improvement")

        # Get performance data
        perf = agent_pool.get_agent_performance(candidate.agent_id) if candidate.agent_id else None
        if not perf:
            raise ValueError(f"No performance data for candidate {candidate.id}")

        # Step 1: Load trajectory files
        trajectory_files = self._load_trajectory_files(trajectory_path)
        if not trajectory_files:
            raise ValueError(f"No trajectory files found at: {trajectory_path}")
        logger.info(f"Found {len(trajectory_files)} trajectory files for reflection analysis")

        # Step 2: Extract game data for this candidate
        game_data_list = self._extract_game_data_from_trajectories(trajectory_files, candidate)
        if not game_data_list:
            raise ValueError(f"No game data for {candidate.id} in trajectories")

        wins = sum(1 for g in game_data_list if g['result'] == 'win')
        losses = sum(1 for g in game_data_list if g['result'] == 'loss')
        draws = sum(1 for g in game_data_list if g['result'] == 'draw')
        logger.info(f"Reflecting on {len(game_data_list)} games for {candidate.id} (W:{wins}, L:{losses}, D:{draws})")

        # Step 3: Generate reflections
        reflections = []
        for game_data in game_data_list:
            r = self._reflect_on_trajectory(game_data, candidate.prompt, game_data['result'])
            reflections.append({
                'result': game_data['result'], 'reflection': r['reflection'],
                'insight': r['insight'], 'format_errors': game_data['format_errors'],
                'total_moves': game_data['total_moves']
            })

        if not reflections:
            raise ValueError(f"Failed to generate reflections from {len(game_data_list)} games")

        # Step 4: Synthesize reflections
        overall_stats = {'win_rate': perf.win_rate(), 'wins': perf.wins, 'losses': perf.losses, 'draws': perf.draws}
        synthesis, reflections = self._synthesize_reflections(reflections, overall_stats, candidate.prompt)
        candidate.synthesis = synthesis
        logger.info(f"Synthesized {len(reflections)} reflections for {candidate.id}")

        # Step 5: Generate improved prompt
        improvement_prompt = f"""You are an expert at improving game-playing prompts based on performance data and strategic analysis.

CURRENT PROMPT: {candidate.prompt}

PERFORMANCE DATA:
- Win Rate: {perf.win_rate():.1%} ({perf.wins}W / {perf.losses}L / {perf.draws}D)
- Total Games: {perf.games_played}
- TrueSkill: \u03bc={perf.trueskill_rating.mu:.1f}

STRATEGIC SYNTHESIS FROM GAME ANALYSIS:
{synthesis}

YOUR TASK:
Create an improved prompt for {self.env_id} that will achieve better performance based on the analysis above.

Provide your response in valid JSON format:

{{
  "thinking": "Your analysis: (1) What's working, (2) What weaknesses exist, (3) Strategy for improvement, (4) How changes will help",
  "final_prompt": "The improved prompt."
}}

Ensure your response is valid JSON with these exact two fields."""

        response = self.analyzer(improvement_prompt)
        result = self._parse_json_response(response)

        improved = result['final_prompt'].strip()
        if result['thinking']:
            logger.debug(f"Improvement thinking for {candidate.id}: {result['thinking'][:200]}...")

        if hasattr(candidate, 'synthesis'):
            candidate.avg_reflection_importance = 3.0

        if not improved or len(improved) < 10:
            raise ValueError(f"Generated prompt too short: {improved}")

        logger.info(f"Generated improved prompt for {candidate.id}: {improved[:100]}...")
        return improved

    def _crossover(self, parent1: PromptCandidate, parent2: PromptCandidate) -> str:
        """Crossover two parent prompts.
        
        PLACEHOLDER: To be implemented with sophisticated crossover
        """
        if self.analyzer:
            prompt = f"""Combine the best elements of these two {self.env_id} prompts:

Parent 1: {parent1.prompt}
Parent 2: {parent2.prompt}

Create a hybrid that:
1. Integrates strengths from both
2. Maintains coherent strategy
3. Stays concise (1-3 sentences)

Hybrid prompt:"""
            
            response = self.analyzer(prompt)
            return response.strip().strip('"')
        
        # Simple fallback: combine halves
        words1 = parent1.prompt.split()
        words2 = parent2.prompt.split()
        
        # Take first half of parent1 and second half of parent2
        mid1 = len(words1) // 2
        mid2 = len(words2) // 2
        
        combined = words1[:mid1] + words2[mid2:]
        return " ".join(combined)
    
    def get_best_candidate(self, agent_pool, fitness_method: str = "trueskill") -> Optional[PromptCandidate]:
        """Get the best performing candidate based on AgentPool data.
        
        Args:
            agent_pool: AgentPool containing performance data
            fitness_method: Method to rank agents - "trueskill" or "winrate" (default: "trueskill")
            
        Returns:
            Best performing PromptCandidate or None if no population
        """
        if not self.population:
            return None
        
        # Validate fitness_method
        assert fitness_method in ["trueskill", "winrate"], f"fitness_method must be 'trueskill' or 'winrate', got '{fitness_method}'"
        
        # Get rankings from pool
        rankings = agent_pool.get_ranked_agents(fitness_method=fitness_method)
        
        # Find best candidate
        best_fitness = -float('inf')
        best_candidate = None
        
        for candidate in self.population:
            if candidate.agent_id:
                for agent_id, fitness_value in rankings:
                    if agent_id == candidate.agent_id:
                        if fitness_value > best_fitness:
                            best_fitness = fitness_value
                            best_candidate = candidate
                        break
        
        return best_candidate if best_candidate else self.population[0]
    
    def save_generation(self, agent_pool=None):
        """Save current generation state."""
        gen_dir = self.output_dir / f"generation_{self.generation:02d}"
        gen_dir.mkdir(exist_ok=True)
        
        # Prepare population data
        population_data = []
        for candidate in self.population:
            data = candidate.to_dict()
            
            # Add performance if available
            if agent_pool and candidate.agent_id:
                perf = agent_pool.get_agent_performance(candidate.agent_id)
                if perf:
                    data["performance"] = {
                        "trueskill_mu": perf.trueskill_rating.mu,
                        "trueskill_sigma": perf.trueskill_rating.sigma,
                        "win_rate": perf.win_rate(),
                        "games_played": perf.games_played,
                        "wins": perf.wins,
                        "losses": perf.losses,
                        "draws": perf.draws
                    }
            
            population_data.append(data)
        
        with open(gen_dir / "population.json", 'w') as f:
            json.dump(population_data, f, indent=2)
        
        logger.info(f"Saved generation {self.generation} to {gen_dir}")
    
    def print_summary(self, agent_pool=None):
        """Print generation summary."""
        print(f"Training: Generation {self.generation} Summary")
        print("-" * 80)
        
        if agent_pool:
            # Get rankings using the engine's fitness method
            rankings = agent_pool.get_ranked_agents(fitness_method=self.fitness_method)
            ranking_dict = dict(rankings)
            
            # Sort candidates by fitness score
            candidates_with_fitness = []
            for candidate in self.population:
                if candidate.agent_id and candidate.agent_id in ranking_dict:
                    fitness_score = ranking_dict[candidate.agent_id]
                    perf = agent_pool.get_agent_performance(candidate.agent_id)
                    candidates_with_fitness.append((candidate, fitness_score, perf))
            
            candidates_with_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Always show TrueSkill in display, but sort by configured fitness method
            print(f"{'Rank':<5} {'ID':<45} {'Method':<12} {'TrueSkill':<10} {'Win%':<8} {'Loss%':<8} {'Draw%':<8}")
            print("-" * 105)
            
            for i, (candidate, fitness_score, perf) in enumerate(candidates_with_fitness[:10]):
                win_rate = perf.win_rate() if perf else 0
                loss_rate = perf.losses / perf.games_played if perf and perf.games_played > 0 else 0
                draw_rate = perf.draws / perf.games_played if perf and perf.games_played > 0 else 0
                
                # Get TrueSkill score for display (regardless of ranking method)
                trueskill_score = perf.trueskill_rating.mu if perf else 25.0
                
                # Show elite lineage info
                display_id = candidate.id
                if candidate.creation_method == "elite" and candidate.parent_id:
                    display_id = f"{candidate.id} (from {candidate.parent_id})"
                
                print(f"{i+1:<5} {display_id:<45} {candidate.creation_method:<12} "
                      f"{trueskill_score:<10.2f} {win_rate:<8.1%} {loss_rate:<8.1%} {draw_rate:<8.1%}")
            
            if candidates_with_fitness:
                best = candidates_with_fitness[0][0]
                print(f"\nBest: {best.id}")
                print(f"   Prompt: {best.prompt[:100]}...")
        else:
            # Just show candidates
            print(f"{'ID':<30} {'Method':<12} {'Parent':<20}")
            print("-" * 80)
            
            for candidate in self.population[:10]:
                parent = candidate.parent_id or "None"
                print(f"{candidate.id:<30} {candidate.creation_method:<12} {parent:<20}")