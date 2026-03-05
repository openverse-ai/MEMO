"""
Trajectory-based Memory System for Self-Play Tournament Learning
Analyzes trajectory folders and generates insights for memory agents.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

import textarena as ta
import wandb

from .prompts import (
    # Memory merge prompts (basic)
    XML_CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE, XML_CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE,
    # Abstract merge prompts (basic)
    XML_CRUD_STATE_ABSTRACT_V2_PROMPT_TEMPLATE, XML_CRUD_STATE_ABSTRACT_V2_FORMAT_TEMPLATE,
)
from .xml_crud_operations import XMLCRUDParser

logger = logging.getLogger(__name__)


@dataclass
class CompressedMove:
    """Compressed representation of a move for analysis with full context."""
    step: int
    player_id: int  # Which player made this move (0 or 1)
    action: str
    raw_action: str  # Full agent response before action extraction
    format_correct: bool
    invalid_move: bool = False
    abstract_state: str = ""  # Abstract game state (opening/midgame/endgame/advantage/disadvantage)
    game_context: str = ""   # Specific context for this move
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompressedGame:
    """Compressed game representation for efficient analysis."""
    game_id: str
    agent_name: str
    player_id: int
    outcome: float  # 1=win, -1=loss, 0=draw
    moves: List[CompressedMove]
    total_format_errors: int
    total_invalid_moves: int
    game_length: int
    
    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "agent_name": self.agent_name,
            "player_id": self.player_id,
            "outcome": self.outcome,
            "moves": [m.to_dict() for m in self.moves],
            "total_format_errors": self.total_format_errors,
            "total_invalid_moves": self.total_invalid_moves,
            "game_length": self.game_length
        }


class StateRegistry:
    """Maintains a registry of known game states."""
    
    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or Path("state_registry.json")
        self.known_states = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load existing state registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        
        # Default states if no registry exists
        return {
            "opening_balanced_normal": {
                "description": "Early game with balanced positions",
                "keywords": ["opening", "start", "early", "initial", "balanced"],
                "examples": ["Game just started", "First few moves"]
            },
            "midgame_balanced_normal": {
                "description": "Middle game with roughly equal chances",
                "keywords": ["middle", "midgame", "developing", "balanced"],
                "examples": ["Pieces developed", "No clear advantage"]
            },
            "endgame_winning_critical": {
                "description": "Late game with winning opportunity",
                "keywords": ["endgame", "winning", "advantage", "critical"],
                "examples": ["Clear material advantage", "Winning position"]
            }
        }
    
    def save_registry(self):
        """Save state registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.known_states, f, indent=2)
    
    def find_similar_state(self, description: str, keywords: List[str]) -> Optional[str]:
        """Find similar existing state based on keywords and description."""
        max_similarity = 0
        best_match = None
        
        for state_id, state_data in self.known_states.items():
            # Calculate keyword overlap
            state_keywords = set(state_data["keywords"])
            query_keywords = set(keywords)
            keyword_overlap = len(state_keywords & query_keywords)
            
            # Simple text similarity (can be improved)
            desc_words = set(description.lower().split())
            state_desc_words = set(state_data["description"].lower().split())
            desc_overlap = len(desc_words & state_desc_words)
            
            similarity = keyword_overlap * 2 + desc_overlap
            if similarity > max_similarity and similarity >= 3:  # Threshold for similarity
                max_similarity = similarity
                best_match = state_id
        
        return best_match
    
    def add_new_state(self, state_id: str, description: str, keywords: List[str], example: str):
        """Add a new state to the registry."""
        self.known_states[state_id] = {
            "description": description,
            "keywords": keywords,
            "examples": [example],
            "created": datetime.now().isoformat()
        }
        self.save_registry()
        logger.info(f"Added new state: {state_id} - {description}")


class StateClassifier:
    """Classifies game states with growing state registry."""
    
    def __init__(self, model_name: str = "google/gemini-2.0-flash-001", registry_file: Optional[Path] = None):
        # Try to use OpenAIOpenrouterAgent if available
        try:
            from textarena.agents import OpenAIOpenrouterAgent
            self.classifier = OpenAIOpenrouterAgent(
                model_name=model_name,
                system_prompt=self._get_classifier_prompt()
            )
        except ImportError:
            self.classifier = ta.agents.OpenRouterAgent(
                model_name=model_name,
                system_prompt=self._get_classifier_prompt(),
                contribute_to_optimization=True  # Disable optimization to avoid issues
            )
        self.registry = StateRegistry(registry_file)
    
    def _get_classifier_prompt(self) -> str:
        return """You are a game state classifier. Analyze game observations and classify them using existing known states or create new ones.

RESPOND with a JSON object containing:
{
    "description": "Detailed description of the current game state",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "state_category": "suggested_state_name_if_new",
    "confidence": 0.8
}

Focus on strategic and tactical aspects rather than just turn numbers."""
    
    def classify_state(self, observation: str) -> Dict[str, str]:
        """Classify game state using growing registry system."""
        # Get LLM analysis of the current state
        response = self.classifier(f"Analyze this game state:\n{observation}")
        
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if not json_match:
            return self._fallback_classification()
        
        analysis = json.loads(json_match.group())
        description = analysis["description"]
        keywords = analysis["keywords"]
        
        # Try to find similar existing state
        similar_state = self.registry.find_similar_state(description, keywords)
        
        if similar_state:
            # Use existing state
            state_data = self.registry.known_states[similar_state]
            return {
                "state_id": similar_state,
                "description": state_data["description"],
                "keywords": state_data["keywords"],
                "matched_existing": True
            }
        else:
            # Create new state
            state_category = analysis["state_category"]
            confidence = analysis["confidence"]
            
            if confidence > 0.6:  # Only create new states if confident
                # Generate unique state ID
                state_id = f"{state_category}_{len(self.registry.known_states)}"
                
                # Add to registry
                self.registry.add_new_state(
                    state_id, 
                    description, 
                    keywords,
                    observation[:100]  # Example
                )
                
                return {
                    "state_id": state_id,
                    "description": description,
                    "keywords": keywords,
                    "matched_existing": False
                }
            else:
                # Low confidence, use fallback
                return self._fallback_classification()
    
    def _fallback_classification(self) -> Dict[str, str]:
        """Fallback classification when analysis fails."""
        return {
            "state_id": "midgame_balanced_normal",
            "description": "Standard middle game situation",
            "keywords": ["midgame", "standard", "balanced"],
            "matched_existing": True
        }


class TrajectoryAnalyzer:
    """Analyzes trajectory files and generates insights."""
    
    def __init__(self, model_name: str = "google/gemini-2.0-flash-001", prompt_style: str = "basic", max_games_in_prompt: int = 1, game_sequence_style: str = "raw_action_only"):
        """Initialize analyzer with LLM."""
        # Try to use OpenAIOpenrouterAgent if available
        try:
            from textarena.agents import OpenAIOpenrouterAgent
            self.analyzer = OpenAIOpenrouterAgent(
                model_name=model_name,
                system_prompt=self._get_analyzer_prompt()
            )
        except ImportError:
            self.analyzer = ta.agents.OpenRouterAgent(
                model_name=model_name,
                system_prompt=self._get_analyzer_prompt(),
                contribute_to_optimization=True
            )
        # Only initialize state classifier if not using simple prompt
        if prompt_style != "simple":
            self.state_classifier = StateClassifier(model_name)
        else:
            self.state_classifier = None
        self.prompt_style = prompt_style  # "basic", "detailed", or "simple"
        self.max_games_in_prompt = max_games_in_prompt  # Max games to show in detailed/simple prompt
        self.game_sequence_style = game_sequence_style  # How to format game sequences
        
        # Prompts directory will be set by the memory system
        self.prompts_dir = Path("analysis_prompts")  # Default fallback
    
    def _get_analyzer_prompt(self) -> str:
        """Get analysis prompt for the LLM."""
        return """You are a game strategy analyst. Analyze compressed game data to identify patterns and provide insights."""
    
    def _get_action_text(self, move: CompressedMove) -> str:
        """Get action text based on game_sequence_style."""
        if self.game_sequence_style == "raw_action_only":
            return move.raw_action
        else:
            # Future sequence styles can be added here
            return move.action
    
    def compress_trajectory_file(self, trajectory_file: Path, max_trajectories: int = -1) -> List[CompressedGame]:
        """Compress a trajectory JSON file into essential information.
        
        Args:
            trajectory_file: Path to trajectory JSON file
            max_trajectories: Maximum number of trajectories to process (-1 = use all)
        """
        with open(trajectory_file, 'r') as f:
            data = json.load(f)
        
        # Limit trajectories if specified
        if max_trajectories > 0 and len(data) > max_trajectories:
            logger.info(f"Limiting trajectories from {len(data)} to {max_trajectories} for {trajectory_file.name}")
            data = data[:max_trajectories]
        
        compressed_games = []
        
        for game_data in data:
            # Process each agent in the game  
            # Required keys: use direct access (fail fast)
            agent_names = game_data["agent_names"] 
            rewards = game_data["rewards"]
            trajectory = game_data["trajectory"]
            
            # Group moves by agent
            agent_moves = defaultdict(list)
            for step in trajectory:
                # Direct access to required keys (will raise KeyError if missing)
                pid = step["player_id"]
                agent_name = step["agent_name"]
                
                # Classify game state for this move using growing state system (skip for simple prompt)
                observation = step["observation"]
                if self.prompt_style != "simple" and self.state_classifier:
                    state_info = self.state_classifier.classify_state(observation) if observation else {
                        "state_id": "unknown_state", "description": "Unknown game state", "keywords": ["unknown"]
                    }
                    abstract_state = state_info["state_id"]
                    game_context = state_info["description"]
                else:
                    # Skip state classification for simple prompt
                    abstract_state = ""
                    game_context = ""
                
                # Extract key information with direct access and temporal context
                # Required keys use direct access (fail fast if missing)
                # Optional keys in step_info use .get() since step_info can be empty {}
                
                move = CompressedMove(
                    step=step["step"],  # Required: direct access
                    player_id=pid,  # Which player made this move
                    action=step["action"],  # Required: direct access  
                    raw_action=step["raw_action"],  # Required: direct access to full agent response
                    format_correct=step["format_feedback"]["correct_answer_format"],  # Required: direct access
                    invalid_move=step["step_info"].get("invalid_move", False),  # Optional: step_info can be empty
                    abstract_state=abstract_state,
                    game_context=game_context
                )
                agent_moves[agent_name].append(move)

            
            # Create compressed game for each agent
            for i, agent_name in enumerate(agent_names):
                if i < len(rewards):
                    outcome = 1 if rewards[i] > 0 else (-1 if rewards[i] < 0 else 0)
                    moves = agent_moves[agent_name] if agent_name in agent_moves else []
                    
                    # Calculate error counts
                    format_errors = sum(1 for m in moves if not m.format_correct)
                    invalid_moves = sum(1 for m in moves if m.invalid_move)
                    
                    compressed_game = CompressedGame(
                        game_id=game_data["game_id"],  # Required: direct access
                        agent_name=agent_name,
                        player_id=i,
                        outcome=outcome,
                        moves=moves,
                        total_format_errors=format_errors,
                        total_invalid_moves=invalid_moves,
                        game_length=len(moves)
                    )
                    compressed_games.append(compressed_game)
        
        return compressed_games
    
    def _create_basic_prompt(self, games: List[CompressedGame], wins: List[CompressedGame], losses: List[CompressedGame], draws: List[CompressedGame], winning_moves: List[str], losing_moves: List[str], format_issues: List[str], invalid_issues: List[str]) -> str:
        """Create basic prompt (original style) adapted for self-play analysis."""
        # Combine all moves since in self-play both players are the same agent
        all_moves = winning_moves + losing_moves
        
        prompt = f"""You are analyzing SELF-PLAY game performance to extract ABSTRACT strategic principles.

SELF-PLAY GAME STATISTICS:
- Total games: {len(games)} (both players are identical agents)
- Agent outcomes: {len(wins)} wins, {len(losses)} losses, {len(draws)} draws
- Average format errors per game: {sum(g.total_format_errors for g in games) / len(games):.1f}
- Average invalid moves per game: {sum(g.total_invalid_moves for g in games) / len(games):.1f}

MOVE SAMPLES FROM SELF-PLAY: {all_moves[:20]}
FORMAT ERROR EXAMPLES: {format_issues[:5]}
INVALID MOVE EXAMPLES: {invalid_issues[:5]}

Extract GENERAL strategic principles (not specific moves) in JSON format:
{{
    "successful_patterns": ["abstract_principle1", "abstract_principle2", "abstract_principle3"],
    "failure_patterns": ["abstract_mistake1", "abstract_mistake2", "abstract_mistake3"],
    "improvement_suggestions": ["general_advice1", "general_advice2", "general_advice3"],
    "strategic_advice": "High-level strategic guidance",
    "common_format_issues": ["formatting_principle1", "formatting_principle2"]
}}

Focus on DETAILED ABSTRACT principles like:
- "Control central positions early to establish board dominance"
- "Avoid repetitive patterns that allow opponents to predict your strategy"  
- "Respond dynamically to opponent pressure while maintaining your position"
- "Maintain move diversity across opening, mid-game, and closing phases"
- "Balance aggressive positioning with defensive contingencies"
- "Exploit opponent weaknesses when they over-commit to predictable sequences"
- "Adapt your strategy based on opponent's playing style and tendencies"

IMPORTANT: Both players are identical agents, so analyze patterns across all game positions rather than separating winners/losers.
Provide ACTIONABLE strategic guidance, NOT specific moves like "[4]" or "[2]"."""
        
        return prompt
    
    def _create_complete_trajectory_sequences(self, games: List[CompressedGame], max_games: int = 1) -> List[str]:
        """Create complete trajectory sequences showing the entire game process step by step."""
        sequences = []
        game_count = 0
        
        # Group games by game_id to reconstruct full games (same approach as _create_game_sequences)
        games_by_id = {}
        for game in games:
            if game.game_id not in games_by_id:
                games_by_id[game.game_id] = []
            games_by_id[game.game_id].append(game)
        
        # Process each complete game (needs both agents' data)
        for game_id, agent_games in games_by_id.items():
            if game_count >= max_games:
                break
                
            # Skip incomplete games (need both agents for complete trajectory)
            if len(agent_games) < 2:
                continue
                
            game_count += 1
            
            # Combine moves from both agents
            all_moves_by_step = {}
            for agent_game in agent_games:
                for move in agent_game.moves:
                    if move.step not in all_moves_by_step:
                        all_moves_by_step[move.step] = {}
                    all_moves_by_step[move.step][move.player_id] = move
            
            # Build complete game trajectory
            sequence_parts = [f"\n=== GAME {game_count} COMPLETE TRAJECTORY ==="]
            sequence_parts.append(f"Game ID: {game_id}")
            
            for step in sorted(all_moves_by_step.keys()):
                sequence_parts.append(f"\n--- STEP {step} ---")
                step_moves = all_moves_by_step[step]
                
                # Show moves for each player in this step
                for pid in sorted(step_moves.keys()):
                    move = step_moves[pid]
                    action_text = self._get_action_text(move)
                    
                    # Add action with direct error labels
                    error_labels = []
                    if not move.format_correct:
                        error_labels.append("[FORMAT ERROR]")
                    if move.invalid_move:
                        error_labels.append("[INVALID MOVE]")
                    
                    error_suffix = " " + " ".join(error_labels) if error_labels else ""
                    sequence_parts.append(f"Player {pid} Action: {action_text}{error_suffix}")
            
            # Add game result (from perspective of first agent)
            if agent_games[0].outcome > 0:
                result_desc = f"GAME RESULT: Player {agent_games[0].player_id} (Agent {agent_games[0].agent_name}) WON"
            elif agent_games[0].outcome < 0:
                result_desc = f"GAME RESULT: Player {agent_games[0].player_id} (Agent {agent_games[0].agent_name}) LOST"
            else:
                result_desc = "GAME RESULT: DRAW"
            
            sequence_parts.append(f"\n{result_desc}")
            
            # Add stats for both agents
            for i, agent_game in enumerate(agent_games):
                sequence_parts.append(f"Player {agent_game.player_id} ({agent_game.agent_name}): {len(agent_game.moves)} moves, {agent_game.total_format_errors} format errors, {agent_game.total_invalid_moves} invalid moves")
            
            sequence_parts.append("=" * 50)
            
            sequences.append('\n'.join(sequence_parts))
        
        return sequences

    def _create_detailed_prompt(self, games: List[CompressedGame], wins: List[CompressedGame], losses: List[CompressedGame], draws: List[CompressedGame]) -> str:
        """Create detailed prompt showing complete trajectory integrity for self-play analysis."""
        # For self-play, we analyze all games together since both players are identical agents
        all_game_trajectories = self._create_complete_trajectory_sequences(games, max_games=self.max_games_in_prompt)
        
        prompt = f"""SELF-PLAY TRAJECTORY ANALYSIS - COMPLETE GAME INTEGRITY PRESERVED

GAME STATISTICS:
- Total self-play games analyzed: {len(games)}
- Agent outcomes: {len(wins)} wins, {len(losses)} losses, {len(draws)} draws
- Note: In self-play, both players are identical agents, so wins/losses represent different game positions

COMPLETE SELF-PLAY GAME TRAJECTORIES (showing entire decision-making process):
{chr(10).join(all_game_trajectories) if all_game_trajectories else "No games to analyze"}

Based on the COMPLETE self-play trajectories above (showing every step, observation, and decision from both identical agents), extract strategic insights in JSON format:
{{
    "successful_patterns": ["pattern1", "pattern2", "pattern3"],
    "failure_patterns": ["mistake1", "mistake2", "mistake3"],
    "improvement_suggestions": ["advice1", "advice2", "advice3"],
    "strategic_advice": "High-level strategic guidance based on complete self-play analysis",
    "common_format_issues": ["issue1", "issue2"]
}}

IMPORTANT: 
- Analyze the complete decision-making process across all game positions
- Focus on general strategic patterns rather than win/loss specific strategies
- Every step, observation, and decision is preserved to maintain full game integrity"""
        
        return prompt
    
    def _create_simple_prompt(self, games: List[CompressedGame], wins: List[CompressedGame], losses: List[CompressedGame], draws: List[CompressedGame]) -> str:
        """Create simple prompt with complete trajectories and think-answer format."""
        all_game_trajectories = self._create_complete_trajectory_sequences(games, max_games=self.max_games_in_prompt)
        
        prompt = f"""Analyze these self-play game trajectories to extract strategic insights.

GAME STATISTICS:
- Total games: {len(games)} (wins: {len(wins)}, losses: {len(losses)}, draws: {len(draws)})

COMPLETE GAME TRAJECTORIES:
{chr(10).join(all_game_trajectories) if all_game_trajectories else "No games to analyze"}

Focus on:
1. What moves led to wins vs losses
2. Format and invalid move issues  
3. Strategic patterns that work or fail
4. Concrete improvement suggestions

Be specific and actionable.

IMPORTANT: Format your response as follows:
<think>
[Your detailed analysis and reasoning about the game patterns]
</think>

<answer>
1. [First key insight from a strategic perspective]
2. [Second key insight from an error/format perspective]
3. [Third key insight from an improvement perspective]
</answer>"""
        
        return prompt
    
    def _parse_simple_response(self, response: str, max_retries: int = 5) -> Dict[str, Any]:
        """Parse simple format response with think-answer tags and retry on format errors."""
        import re
        
        for attempt in range(max_retries):
            # Extract think and answer sections
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            
            if not think_match or not answer_match:
                if attempt < max_retries - 1:
                    logger.warning(f"Simple format parsing failed (attempt {attempt + 1}). Missing tags. Retrying...")
                    # Retry with explicit format reminder
                    retry_prompt = f"""Your previous response was missing the required format tags.
Please reformat your response EXACTLY as:
<think>
[Your analysis here]
</think>
<answer>
1. [First insight]
2. [Second insight]
3. [Third insight]
</answer>
Previous response: {response}"""
                    response = self.analyzer(retry_prompt)
                    continue
                else:
                    raise ValueError(f"Failed to parse simple format after {max_retries} attempts. Response missing <think> or <answer> tags.")
            
            think_content = think_match.group(1).strip()
            answer_content = answer_match.group(1).strip()
            
            # Parse answer into 3 insights
            insights = []
            for line in answer_content.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):  # Lines starting with number
                    insight = re.sub(r'^\d+\.\s*', '', line).strip()
                    if insight:
                        insights.append(insight)
            
            if len(insights) < 3:
                if attempt < max_retries - 1:
                    logger.warning(f"Simple format parsing failed (attempt {attempt + 1}). Found {len(insights)} insights, need 3. Retrying...")
                    retry_prompt = f"""Your answer section must contain EXACTLY 3 numbered insights.

<answer>
1. [Strategic insight]
2. [Error/format insight]
3. [Improvement insight]
</answer>

You provided: {answer_content}"""
                    response = self.analyzer(retry_prompt)
                    continue
                else:
                    # Pad with empty insights if needed
                    while len(insights) < 3:
                        insights.append("No additional insight provided")
            
            # Simple format - just store raw insights directly
            return {
                "insights": insights,  # Raw list of insights from model
                "format": "simple"
            }
        
        # Should not reach here
        raise ValueError("Failed to parse simple format response")
    
    def analyze_compressed_games(self, agent_name: str, games: List[CompressedGame]) -> Dict[str, Any]:
        """Analyze compressed games to extract strategic insights with prompt logging."""
        if not games:
            return {"error": "No games to analyze"}
        
        # Store games for CRUD operations
        self._last_analyzed_games = games
        
        # Create prompts directory for logging - automatically set in constructor
        self.prompts_dir.mkdir(exist_ok=True, parents=True)
        
        # Separate wins and losses
        wins = [g for g in games if g.outcome > 0]
        losses = [g for g in games if g.outcome < 0]
        draws = [g for g in games if g.outcome == 0]
        
        # Extract key patterns from winning games
        winning_moves = []
        for game in wins[:5]:  # Sample first 5 wins
            for move in game.moves:
                if move.format_correct and not move.invalid_move:
                    winning_moves.append(self._get_action_text(move))
        
        # Extract failure patterns from losing games
        losing_moves = []
        format_issues = []
        invalid_issues = []
        for game in losses[:5]:  # Sample first 5 losses
            for move in game.moves:
                if not move.format_correct:
                    format_issues.append(self._get_action_text(move))
                if move.invalid_move:
                    invalid_issues.append(self._get_action_text(move))
                losing_moves.append(self._get_action_text(move))
        
        # Create analysis prompt based on style
        if self.prompt_style == "basic":
            prompt = self._create_basic_prompt(games, wins, losses, draws, winning_moves, losing_moves, format_issues, invalid_issues)
        elif self.prompt_style == "detailed":
            prompt = self._create_detailed_prompt(games, wins, losses, draws)
        elif self.prompt_style == "simple":
            prompt = self._create_simple_prompt(games, wins, losses, draws)
        else:
            raise ValueError(f"Invalid prompt style: {self.prompt_style}")
        
        # Log the prompt for analysis with generation info if available
        generation = self.current_generation
        
        # Create different log structure for simple format
        if self.prompt_style == "simple":
            prompt_log = {
                "generation": generation,
                "agent_name": agent_name,
                "total_games": len(games),
                "wins": len(wins),
                "losses": len(losses),
                "prompt": prompt,
                "format": "simple",
                "timestamp": datetime.now().isoformat()
            }
        else:
            prompt_log = {
                "generation": generation,
                "agent_name": agent_name,
                "total_games": len(games),
                "wins": len(wins),
                "losses": len(losses),
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }
        prompt_filename = f"analysis_prompt_gen{generation}.json"
        prompt_file = self.prompts_dir / prompt_filename
        with open(prompt_file, 'w') as f:
            json.dump(prompt_log, f, indent=2)
        
        # Log prompt length to wandb
        prompt_length = len(prompt)
        if wandb.run is not None:  # Only log if wandb is initialized
            # Use generation number as step for consistent time-series tracking
            # All generation-level metrics must use step=generation to avoid conflicts
            wandb.log({
                "memory_analysis/prompt_length": prompt_length,
                "memory_analysis/total_games_analyzed": len(games),
            }, step=generation)
            
            # Log prompt style as config (not time-series metric)
            wandb.config.update({"prompt_style": self.prompt_style})
        
        response = self.analyzer(prompt)
        
        # Handle different response formats based on prompt style
        if self.prompt_style == "simple":
            # Parse think-answer format
            parsed = self._parse_simple_response(response)
        else:
            # Parse JSON response - fail fast if invalid
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if not json_match:
                raise ValueError(f"LLM response does not contain valid JSON format. Response: {response[:500]}...")
            
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM response contains invalid JSON: {e}. JSON string: {json_match.group()}") from e
            
            # Validate required keys are present
            required_keys = ["successful_patterns", "failure_patterns", "improvement_suggestions", "common_format_issues", "strategic_advice"]
            missing_keys = [key for key in required_keys if key not in parsed]
            if missing_keys:
                raise KeyError(f"LLM response missing required keys: {missing_keys}. Got keys: {list(parsed.keys())}")
        
        return parsed
    


class TrajectoryMemorySystem:
    """Main system for managing trajectory-based memory.
    
    Memory Merge Styles:
    - "simple_add": Accumulates insights by adding new to old without modification
    - "no_merge": Uses only new insights, discarding old memory
    - "basic": XML-based CREATE, UPDATE, DELETE with strategic focus:
        * Uses XML format for operations (robust parsing via XMLCRUDParser)
        * Emphasizes high-level strategic principles over tactical specifics
        * Focuses on meta-strategies and transferable knowledge
    
    Replay Merge Styles (for state abstracts):
    - "simple_add": Simple accumulation with state-based deduplication
    - "basic": XML-based CREATE, UPDATE, DELETE for state abstracts
    """
    
    def __init__(
        self,
        memory_merge_style: str,
        insights_dir: str = "insights",
        analyzer_model: str = "google/gemini-2.0-flash-001",
        use_state_memory: bool = True,
        prompt_style: str = "basic",
        max_games_in_prompt: int = 1,
        game_sequence_style: str = "raw_action_only",
        replay_merge_style: str = "simple_add",
        prompt_debug: bool = False
    ):
        """Initialize trajectory memory system.
        
        Args:
            memory_merge_style: How to merge memories - "simple_add", "no_merge", or "basic" (required, no default)
            insights_dir: Directory to save memory insights
            analyzer_model: Model to use for trajectory analysis
            use_state_memory: Whether to generate and use state-specific memory (default: True)
            prompt_style: Analysis prompt style - "basic", "detailed", or "simple" (default: "basic")
            max_games_in_prompt: Max games to show in detailed prompt style (default: 1)
            game_sequence_style: How to format game sequences - "raw_action_only" uses full agent response, other options use extracted action (default: "raw_action_only")
            replay_merge_style: How to merge state abstracts - "simple_add" or "basic" (default: "simple_add")
            prompt_debug: Whether to enable debug logging (default: False)
        """
        self.insights_dir = Path(insights_dir)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_debug = prompt_debug
        
        self.analyzer = TrajectoryAnalyzer(analyzer_model, prompt_style, max_games_in_prompt, game_sequence_style)
        self.analyzer.prompt_style = prompt_style  # Ensure prompt_style is accessible in analyzer
        self.current_generation = 0
        self.use_state_memory = use_state_memory
        self.prompt_style = prompt_style
        self.max_games_in_prompt = max_games_in_prompt
        self.game_sequence_style = game_sequence_style
        self.memory_merge_style = memory_merge_style
        self.replay_merge_style = replay_merge_style
        
        # Load current generation
        self._load_current_generation()
    
    def set_prompts_dir(self, prompts_dir: str):
        """Set the prompts directory for the analyzer."""
        self.analyzer.prompts_dir = Path(prompts_dir)
    
    def _load_current_generation(self):
        """Load the current generation number."""
        gen_file = self.insights_dir / "current_generation.json"
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                data = json.load(f)
                self.current_generation = data["generation"]
        logger.info(f"Current generation: {self.current_generation}")
    
    def _save_current_generation(self):
        """Save the current generation number."""
        gen_file = self.insights_dir / "current_generation.json"
        with open(gen_file, 'w') as f:
            json.dump({"generation": self.current_generation}, f)
    
    def process_trajectory_folder(self, trajectory_folder: Path, generation: Optional[int] = None) -> Dict[str, Any]:
        """Process trajectory files from specified generation and selfplay only.
        
        Args:
            trajectory_folder: Path to folder containing trajectory files
            generation: Generation number to process (defaults to current_generation)
        """
        if not trajectory_folder.exists():
            raise FileNotFoundError(f"Trajectory folder not found: {trajectory_folder}")
        
        # Use specified generation or default to current generation
        target_generation = generation if generation is not None else self.current_generation
        
        # Find trajectory files for target generation and selfplay only
        selfplay_pattern = f"trajectories_gen{target_generation}_selfplay*.json"
        trajectory_files = list(trajectory_folder.glob(selfplay_pattern))
        if not trajectory_files:
            raise FileNotFoundError(f"No selfplay trajectory files found for generation {target_generation} in {trajectory_folder}. Searched for pattern: {selfplay_pattern}")
        
        logger.info(f"Processing {len(trajectory_files)} selfplay trajectory files for generation {target_generation}")
        
        # Store current generation for consistent wandb logging throughout this method
        process_generation = self.current_generation
        
        # Pass target generation to analyzer for correct prompt logging
        self.analyzer.current_generation = process_generation
        
        # Compress all trajectories
        all_compressed_games = []
        for traj_file in trajectory_files:
            try:
                compressed_games = self.analyzer.compress_trajectory_file(traj_file)
                all_compressed_games.extend(compressed_games)
                logger.info(f"Compressed {len(compressed_games)} games from {traj_file.name}")
            except Exception as e:
                raise RuntimeError(f"Failed to process trajectory file {traj_file}: {e}") from e
        
        logger.info(f"Analyzing {len(all_compressed_games)} selfplay games from generation {target_generation}")
        
        # Analyze ALL games together (no agent separation)
        analysis = self.analyzer.analyze_compressed_games("all_agents", all_compressed_games)
        
        # Calculate overall statistics
        total_games = len(all_compressed_games)
        wins = len([g for g in all_compressed_games if g.outcome > 0])
        losses = len([g for g in all_compressed_games if g.outcome < 0])
        draws = len([g for g in all_compressed_games if g.outcome == 0])
        
        # Create memory structure based on configuration
        shared_memory = {
            "generation": target_generation,
            "timestamp": datetime.now().isoformat(),
            "total_games": total_games,
            "use_state_memory": self.use_state_memory,
            "performance": {
                "overall_win_rate": wins / total_games if total_games > 0 else 0,
                "total_wins": wins,
                "total_losses": losses,
                "total_draws": draws,
                "avg_format_errors": sum(g.total_format_errors for g in all_compressed_games) / total_games if total_games > 0 else 0,
                "avg_invalid_moves": sum(g.total_invalid_moves for g in all_compressed_games) / total_games if total_games > 0 else 0
            }
        }
        
        if self.prompt_style == "simple":
            # Create simple memory structure - directly use insights from analysis
            shared_memory.update({
                "format": "simple",
                "insights": analysis["insights"],  # Raw insights from model
                "merge_prompt": "",  # Empty for initial generation
                "merge_response": ""  # Empty for initial generation
            })
        elif self.use_state_memory and self.prompt_style != "simple":
            # Analyze state-specific patterns (skip for simple prompt)
            state_specific_insights = self._analyze_state_specific_patterns(all_compressed_games)
            
            # Create hierarchical memory bank
            shared_memory.update({
                "format": "hierarchical",
                # General strategies (high-level)
                "general_strategies": {
                    "winning_strategies": analysis["successful_patterns"],
                    "things_to_avoid": analysis["failure_patterns"],
                    "format_improvements": analysis["common_format_issues"],
                    "strategic_advice": analysis["strategic_advice"],
                    "key_insights": analysis["improvement_suggestions"]
                },
                # State-specific strategies (contextual)
                "state_specific_strategies": state_specific_insights
            })
        else:
            # Use flat memory structure (backward compatibility)
            shared_memory.update({
                "format": "flat",
                "winning_strategies": analysis["successful_patterns"],
                "things_to_avoid": analysis["failure_patterns"],
                "format_improvements": analysis["common_format_issues"],
                "strategic_advice": analysis["strategic_advice"],
                "key_insights": analysis["improvement_suggestions"]
            })
        
        # Save shared memory bank with better naming
        shared_file = self.insights_dir / f"memory_gen{target_generation}.json"
        with open(shared_file, 'w') as f:
            json.dump(shared_memory, f, indent=2)
        
        logger.info(f"Generated shared memory bank for generation {target_generation}")
        
        # Log final memory generation summary to wandb
        if wandb.run is not None:
            # Use process_generation as step for consistent tracking with memory analysis
            wandb.log({
                "memory_generation/total_games_processed": total_games,
                "memory_generation/wins": wins,
                "memory_generation/losses": losses,
                "memory_generation/draws": draws,
                "memory_generation/avg_format_errors": sum(g.total_format_errors for g in all_compressed_games) / total_games if total_games > 0 else 0,
                "memory_generation/avg_invalid_moves": sum(g.total_invalid_moves for g in all_compressed_games) / total_games if total_games > 0 else 0,
            }, step=process_generation)
        
        return shared_memory
    
    def _analyze_state_specific_patterns(self, all_games: List[CompressedGame]) -> Dict[str, Any]:
        """Use LLM to analyze patterns specific to different game states."""
        # Safety check: never run state-specific analysis with simple prompt
        if self.prompt_style == "simple":
            logger.warning("State-specific analysis called with simple prompt - returning empty patterns")
            return {}
            
        from collections import defaultdict
        state_patterns = {}
        
        # Group moves by abstract state
        state_moves = defaultdict(list)
        for game in all_games:
            for move in game.moves:
                if move.abstract_state and not move.invalid_move and move.format_correct:
                    state_moves[move.abstract_state].append({
                        'action': self._get_action_text(move),
                        'outcome': game.outcome,
                        'context': move.game_context,
                        'step': move.step
                    })
        
        # Use LLM to analyze each state type
        for state, moves in state_moves.items():
            if len(moves) < 5:  # Skip states with too few examples
                continue
            
            # Prepare data for LLM analysis
            winning_examples = [m for m in moves if m['outcome'] > 0][:8]
            losing_examples = [m for m in moves if m['outcome'] < 0][:8]
            
            analysis_prompt = f"""Analyze this specific game state: {state}

WINNING EXAMPLES in this state:
{chr(10).join([f"- Action: {m['action']}, Context: {m['context']}, Step: {m['step']}" for m in winning_examples])}

LOSING EXAMPLES in this state:
{chr(10).join([f"- Action: {m['action']}, Context: {m['context']}, Step: {m['step']}" for m in losing_examples])}

Provide state-specific strategic insights in JSON format:
{{
    "state_description": "What this game state represents",
    "key_factors": ["factor1", "factor2", "factor3"],
    "recommended_approach": "Strategic approach for this state",
    "common_mistakes": ["mistake1", "mistake2"],
    "success_patterns": ["pattern1", "pattern2"]
}}

Focus on WHY certain moves work better in THIS specific state."""

            # Log state-specific prompt length to wandb
            state_prompt_length = len(analysis_prompt)
            if wandb.run is not None:  # Only log if wandb is initialized
                # Use current_generation for consistent step tracking
                wandb.log({
                    f"state_analysis/prompt_length_{state}": state_prompt_length,
                    f"state_analysis/move_examples_count": len(moves),
                    f"state_analysis/winning_examples": len(winning_examples),
                    f"state_analysis/losing_examples": len(losing_examples)
                }, step=self.current_generation)
            
            response = self.analyzer.analyzer(analysis_prompt)
            # Parse JSON response
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if json_match:
                state_analysis = json.loads(json_match.group())
                state_analysis["total_occurrences"] = len(moves)
                state_analysis["win_rate"] = len(winning_examples) / len(moves) if moves else 0
                state_patterns[state] = state_analysis
            else:
                # Fallback
                state_patterns[state] = {
                    "state_description": f"Game state: {state}",
                    "total_occurrences": len(moves),
                    "win_rate": len(winning_examples) / len(moves) if moves else 0,
                    "recommended_approach": f"Adapt strategy for {state} situations"
                }
        
        return state_patterns
    
    def next_generation(self):
        """Move to the next generation."""
        self.current_generation += 1
        self._save_current_generation()
        logger.info(f"Advanced to generation {self.current_generation}")
    
    def get_shared_memory(self) -> Optional[Dict[str, Any]]:
        """Get the latest shared memory bank."""
        # Look in all_memory subdirectory with correct naming pattern
        all_memory_dir = self.insights_dir / "all_memory"
        
        # Try current generation first
        for gen in range(self.current_generation, -1, -1):
            # Use the correct naming pattern: generation_{XX}_memory.json
            memory_file = all_memory_dir / f"generation_{gen:02d}_memory.json"
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    return json.load(f)
        
        # Backward compatibility: try old naming patterns
        for gen in range(self.current_generation, -1, -1):
            # Try new naming first
            memory_file = self.insights_dir / f"memory_gen{gen}.json"
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    return json.load(f)
            # Fall back to old naming for backward compatibility
            memory_file = self.insights_dir / f"shared_memory_gen{gen}.json"
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    return json.load(f)
        return None
    
    def get_shared_insights(self) -> Optional[Dict[str, Any]]:
        """Get insights from shared memory bank."""
        return self.get_shared_memory()
    
    def update_memory_with_new_trajectories(self, new_trajectory_files: List[Path]) -> Dict[str, Any]:
        """Update memory with new trajectory files and generate next generation memory."""
        logger.info(f"Updating memory with {len(new_trajectory_files)} new trajectory files")
        
        # Load current memory
        current_memory = self.get_shared_memory()
        if not current_memory:
            logger.warning("No existing memory found. Creating new memory from trajectories.")
            if self.prompt_style == "simple":
                current_memory = {
                    "generation": 0,
                    "total_games": 0,
                    "performance": {"overall_win_rate": 0.0, "total_wins": 0, "total_losses": 0, "total_draws": 0, "avg_format_errors": 0.0, "avg_invalid_moves": 0.0},
                    "format": "simple",
                    "use_state_memory": self.use_state_memory,
                    "insights": [],
                    "merge_prompt": "",
                    "merge_response": ""
                }
            else:
                current_memory = {
                    "generation": 0,
                    "total_games": 0,
                    "performance": {"overall_win_rate": 0.0, "total_wins": 0, "total_losses": 0, "total_draws": 0, "avg_format_errors": 0.0, "avg_invalid_moves": 0.0},
                    "format": "flat",
                    "use_state_memory": self.use_state_memory,
                    "winning_strategies": [],
                    "things_to_avoid": [],
                    "format_improvements": [],
                    "strategic_advice": "",
                    "key_insights": []
                }
        
        # Process new trajectories
        all_new_games = []
        for traj_file in new_trajectory_files:
            if traj_file.exists():
                try:
                    compressed_games = self.analyzer.compress_trajectory_file(traj_file)
                    all_new_games.extend(compressed_games)
                    logger.info(f"Processed {len(compressed_games)} games from {traj_file.name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to process trajectory file {traj_file}: {e}") from e
        
        if not all_new_games:
            logger.warning("No new games found to process")
            return current_memory
        
        logger.info(f"Analyzing {len(all_new_games)} new games")
        
        # Store current generation for consistent logging throughout this method
        memory_generation = self.current_generation
        
        # Pass current generation to analyzer for correct prompt logging  
        self.analyzer.current_generation = memory_generation
        
        # Get new insights from new trajectories
        new_analysis = self.analyzer.analyze_compressed_games("all_agents", all_new_games)
        
        # Calculate new performance metrics
        total_games = len(all_new_games)
        wins = len([g for g in all_new_games if g.outcome > 0])
        losses = len([g for g in all_new_games if g.outcome < 0])
        draws = len([g for g in all_new_games if g.outcome == 0])
        
        new_performance = {
            "overall_win_rate": wins / total_games if total_games > 0 else 0,
            "total_wins": wins,
            "total_losses": losses, 
            "total_draws": draws,
            "avg_format_errors": sum(g.total_format_errors for g in all_new_games) / total_games if total_games > 0 else 0,
            "avg_invalid_moves": sum(g.total_invalid_moves for g in all_new_games) / total_games if total_games > 0 else 0
        }
        
        # Combine old and new insights intelligently
        updated_memory = self._combine_memory_insights(current_memory, new_analysis, new_performance, total_games)
        
        # Advance generation
        self.next_generation()
        updated_memory["generation"] = self.current_generation
        updated_memory["timestamp"] = datetime.now().isoformat()
        
        # Save updated memory with better naming
        memory_file = self.insights_dir / f"memory_gen{self.current_generation}.json"
        with open(memory_file, 'w') as f:
            json.dump(updated_memory, f, indent=2)
        
        # Log insights length to wandb using the generation before advancing
        # This ensures consistency with other logs from the same evolutionary generation
        if wandb.run is not None:
            # Calculate insights content length based on memory format
            if updated_memory["format"] == "simple":
                insights_content = "\n".join(updated_memory["insights"])
                insights_length = len(insights_content)
            else:
                # For hierarchical/flat formats, combine all strategic content
                strategic_content = []
                strategic_content.extend(updated_memory["winning_strategies"])
                strategic_content.extend(updated_memory["things_to_avoid"])
                strategic_content.extend(updated_memory["format_improvements"])
                strategic_content.extend(updated_memory["key_insights"])
                if updated_memory["strategic_advice"]:
                    strategic_content.append(updated_memory["strategic_advice"])
                insights_content = "\n".join(strategic_content)
                insights_length = len(insights_content)
            
            wandb.log({
                "memory_persistence/insights_content_length": insights_length,
            }, step=memory_generation)
        
        logger.info(f"Updated memory for generation {self.current_generation}")
        logger.info(f"Combined {current_memory['total_games']} old games with {total_games} new games")
        
        return updated_memory
    
    def _combine_memory_insights(self, old_memory: Dict, new_analysis: Dict, new_performance: Dict, new_games_count: int) -> Dict:
        """Intelligently combine old memory with new insights."""
        
        # Handle simple memory format - if prompt style is simple, always use simple combination
        if self.prompt_style == "simple":
            return self._combine_simple_memory(old_memory, new_analysis, new_performance, new_games_count)
        
        # Use current generation performance (not weighted historical)
        total_games = old_memory["total_games"] + new_games_count
        combined_performance = new_performance  # Show current generation performance to agents
        
        # Combine strategic insights (merge unique items, prioritize newer ones)
        def merge_strategies(old_list: List[str], new_list: List[str], max_items: int = 5) -> List[str]:
            """Merge strategy lists, keeping unique items and prioritizing newer ones."""
            # Start with new strategies (higher priority)
            combined = list(new_list) if new_list else []
            
            # Add old strategies that aren't similar to new ones
            old_list = old_list if old_list else []
            for old_strategy in old_list:
                is_similar = False
                for new_strategy in combined:
                    # Simple similarity check (can be improved)
                    if len(set(old_strategy.lower().split()) & set(new_strategy.lower().split())) > 2:
                        is_similar = True
                        break
                if not is_similar and len(combined) < max_items:
                    combined.append(old_strategy)
            
            return combined[:max_items]
        
        # Combine insights
        combined_memory = {
            "total_games": total_games,
            "performance": combined_performance,
            "winning_strategies": merge_strategies(
                old_memory["winning_strategies"],
                new_analysis["successful_patterns"]
            ),
            "things_to_avoid": merge_strategies(
                old_memory["things_to_avoid"],
                new_analysis["failure_patterns"]
            ),
            "format_improvements": merge_strategies(
                old_memory["format_improvements"],
                new_analysis["common_format_issues"],
                max_items=3
            ),
            "key_insights": merge_strategies(
                old_memory["key_insights"],
                new_analysis["improvement_suggestions"]
            )
        }
        
        # Update strategic advice (prioritize newer advice but keep context)
        old_advice = old_memory["strategic_advice"]
        new_advice = new_analysis["strategic_advice"]
        
        if new_advice and old_advice:
            combined_memory["strategic_advice"] = f"{new_advice} Building on previous experience: {old_advice[:200]}..."
        elif new_advice:
            combined_memory["strategic_advice"] = new_advice
        else:
            combined_memory["strategic_advice"] = old_advice
        
        return combined_memory
    
    def _apply_memory_operations_with_logging(self, old_insights: List[str], new_insights: List[str], memory_merge_style: str) -> Tuple[List[str], str, str, str]:
        """Apply memory operations and return insights, prompt, response, and operation stats.
        
        Args:
            memory_merge_style: Memory merge strategy - supports both regex and XML-based parsing
            
        Returns:
            Tuple of (updated_insights, critique_prompt, llm_response, operation_stats)
        """
        # Build critique prompt for insight operations
        critique_prompt = self._build_insight_critique_prompt(old_insights, new_insights, memory_merge_style=memory_merge_style)
        
        # Get LLM to generate operations
        llm_response = self.analyzer.analyzer(critique_prompt)
        
        # Parse operations from LLM response using XML parsing
        allow_delete = True  # "basic" style supports ADD, EDIT, and REMOVE
        xml_parser = XMLCRUDParser()
        xml_operations = xml_parser.parse_operations(llm_response)
        operations = self._convert_xml_to_legacy_format(xml_operations)
        
        # Count operations by type
        operation_counts = {"ADD": 0, "EDIT": 0, "REMOVE": 0}
        for op_type, _ in operations:
            if "ADD" in op_type:
                operation_counts["ADD"] += 1
            elif "REMOVE" in op_type:
                operation_counts["REMOVE"] += 1
            else:  # EDIT
                operation_counts["EDIT"] += 1
        
        # Format operation stats string
        operation_stats = "; ".join([f"{k}: {v}" for k, v in operation_counts.items() if v > 0])
        if not operation_stats:
            operation_stats = "No operations"
        
        # Apply operations to update insights
        updated_insights = self._update_insights(old_insights, new_insights, operations, allow_delete=allow_delete, memory_merge_style=memory_merge_style)
        
        # Simple logging to wandb if available
        if wandb.run is not None:
            wandb.log({
                "memory_merge/prompt_length": len(critique_prompt),
                "memory_merge/response_length": len(llm_response),
                "memory_merge/operations_count": len(operations),
                "memory_merge/add_count": operation_counts["ADD"],
                "memory_merge/edit_count": operation_counts["EDIT"],
                "memory_merge/remove_count": operation_counts["REMOVE"]
            }, step=self.current_generation)
        
        return updated_insights, critique_prompt, llm_response, operation_stats
    
    def _apply_state_abstract_operations_with_logging(self, old_abstracts: List[Dict], new_abstracts: List[Dict]) -> Tuple[List[Dict], str, str, str]:
        """Apply XML CRUD operations to state abstracts and return updated list with logging info.
        
        Returns:
            Tuple of (updated_abstracts, critique_prompt, llm_response, operation_stats)
        """
        # Build critique prompt for state abstract operations
        critique_prompt = self._build_state_abstract_critique_prompt(old_abstracts, new_abstracts)
        
        # Get LLM to generate operations
        llm_response = self.analyzer.analyzer(critique_prompt)
        
        # Parse operations from LLM response using XML parser
        xml_parser = XMLCRUDParser()
        xml_operations = xml_parser.parse_operations(llm_response)
        operations = self._convert_xml_to_legacy_format(xml_operations)
        
        # Count operations by type
        operation_counts = {"ADD": 0, "EDIT": 0, "REMOVE": 0}
        for op_type, _ in operations:
            if "ADD" in op_type:
                operation_counts["ADD"] += 1
            elif "REMOVE" in op_type:
                operation_counts["REMOVE"] += 1
            else:  # EDIT
                operation_counts["EDIT"] += 1
        
        # Format operation stats string
        operation_stats = "; ".join([f"{k}: {v}" for k, v in operation_counts.items() if v > 0])
        if not operation_stats:
            operation_stats = "No operations"
        
        # Apply operations to update state abstracts
        updated_abstracts = self._update_state_abstracts(old_abstracts, new_abstracts, operations)
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "replay_merge/prompt_length": len(critique_prompt),
                "replay_merge/response_length": len(llm_response),
                "replay_merge/operations_count": len(operations),
                "replay_merge/add_count": operation_counts["ADD"],
                "replay_merge/edit_count": operation_counts["EDIT"],
                "replay_merge/remove_count": operation_counts["REMOVE"]
            }, step=self.current_generation)
        
        return updated_abstracts, critique_prompt, llm_response, operation_stats
    
    def _build_state_abstract_critique_prompt(self, old_abstracts: List[Dict], new_abstracts: List[Dict]) -> str:
        """Build prompt for LLM to critique and suggest operations on state abstracts."""
        # Format old abstracts as numbered list
        if old_abstracts:
            old_abstracts_formatted = []
            for i, abstract_dict in enumerate(old_abstracts, 1):
                # Get the full state_abstract string from the dict
                full_abstract = abstract_dict['state_abstract'] 
                old_abstracts_formatted.append(f"{i}. {full_abstract}")
            old_abstracts_formatted = '\n'.join(old_abstracts_formatted)
        else:
            old_abstracts_formatted = "[EMPTY STATE ANALYSIS LIBRARY]\n\nSince there are no existing state analyses, you can ONLY use ADD operations.\nDo NOT use EDIT or REMOVE operations - there is nothing to edit or remove."
        
        # Format new abstracts as numbered list
        new_abstracts_formatted = []
        for i, abstract_dict in enumerate(new_abstracts, 1):
            # Get the full state_abstract string from the dict
            full_abstract = abstract_dict['state_abstract'] 
            new_abstracts_formatted.append(f"{i}. {full_abstract}")
        new_abstracts_formatted = '\n'.join(new_abstracts_formatted)
        
        # Use basic prompt template
        if self.replay_merge_style == "basic":
            prompt = XML_CRUD_STATE_ABSTRACT_V2_PROMPT_TEMPLATE.format(
                new_abstracts_formatted=new_abstracts_formatted,
                old_abstracts_formatted=old_abstracts_formatted,
                format_template=XML_CRUD_STATE_ABSTRACT_V2_FORMAT_TEMPLATE
            )
        else:
            raise ValueError(f"Invalid replay_merge_style for abstract critique: {self.replay_merge_style}")
        
        return prompt
    
    def _build_insight_critique_prompt(self, old_insights: List[str], new_insights: List[str], memory_merge_style: str) -> str:
        """Build prompt for LLM to critique and suggest operations on insights."""
        # Format old insights as numbered list, with clear guidance when empty
        if old_insights:
            old_insights_formatted = '\n'.join([f"{i}. {insight}" for i, insight in enumerate(old_insights, 1)])
        else:
            old_insights_formatted = "[EMPTY SKILL LIBRARY]\n\nSince there are no existing skills, you can ONLY use ADD operations.\nDo NOT use EDIT or REMOVE operations - there is nothing to edit or remove."
        
        # Format new insights as numbered list
        new_insights_formatted = '\n'.join([f"{i}. {insight}" for i, insight in enumerate(new_insights, 1)])
        
        # Select the appropriate prompt based on memory_merge_style
        if memory_merge_style == "basic":
            prompt = XML_CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE.format(
                new_insights_formatted=new_insights_formatted,
                old_insights_formatted=old_insights_formatted,
                format_template=XML_CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE
            )
        else:
            raise ValueError(f"Invalid memory_merge_style: {memory_merge_style}")
        
        return prompt
    
    def _convert_xml_to_legacy_format(self, xml_operations: List[Tuple[str, str, Optional[int]]]) -> List[Tuple[str, str]]:
        """Convert XML operations to legacy format expected by _update_insights.
        
        Args:
            xml_operations: List of (operation_type, text, insight_number) from XML parser
            
        Returns:
            List of (operation_with_number, text) tuples in legacy format
        """
        legacy_operations = []
        
        for op_type, text, number in xml_operations:
            if op_type == "ADD":
                # For ADD operations, we don't need a number in legacy format
                legacy_operations.append(("ADD", text))
            elif op_type == "EDIT" and number is not None:
                # Format as "EDIT N" for legacy compatibility
                legacy_operations.append((f"EDIT {number}", text))
            elif op_type == "REMOVE" and number is not None:
                # Format as "REMOVE N" for legacy compatibility
                legacy_operations.append((f"REMOVE {number}", text))
            else:
                logger.warning(f"Skipped malformed XML operation: {op_type}, {text}, {number}")
        
        if self.prompt_debug:
            logger.info(f"Converted {len(xml_operations)} XML operations to {len(legacy_operations)} legacy operations")
        return legacy_operations
    
    def _update_insights(self, old_insights: List[str], new_insights: List[str], operations: List[Tuple[str, str]], allow_delete: bool = True, memory_merge_style: str = "") -> List[str]:
        """Apply operations to update insight list (adapted from ExpeL's update_rules)."""
        import re
        
        # Start with a copy of old insights
        updated_insights = list(old_insights)
        
        # Track indices to delete (process in reverse order)
        indices_to_delete = []
        
        # Process operations in order: REMOVE (if allowed), EDIT, ADD (following ExpeL's pattern)
        operation_order = ['REMOVE', 'EDIT', 'ADD'] if allow_delete else ['EDIT', 'ADD']
        
        for op_type in operation_order:
            for operation, text in operations:
                operation_type = operation.split()[0]
                
                if operation_type != op_type:
                    continue
                
                if operation_type == 'REMOVE' and allow_delete:
                    # Extract insight number
                    num_match = re.search(r'REMOVE (\d+)', operation)
                    if num_match:
                        idx = int(num_match.group(1)) - 1  # Convert to 0-based
                        if 0 <= idx < len(updated_insights):
                            indices_to_delete.append(idx)
                
                elif operation_type == 'EDIT':
                    # Extract insight number
                    num_match = re.search(r'EDIT (\d+)', operation)
                    if num_match:
                        idx = int(num_match.group(1)) - 1  # Convert to 0-based
                        if 0 <= idx < len(updated_insights):
                            updated_insights[idx] = text
                
                elif operation_type == 'ADD':
                    # Add new insight if it's not too similar to existing ones (following ExpeL's duplicate detection)
                    is_duplicate = False
                    for existing in updated_insights:
                        # Simple similarity check - if too many overlapping words, consider duplicate
                        existing_words = set(existing.lower().split())
                        new_words = set(text.lower().split())
                        overlap = len(existing_words & new_words)
                        # Use relative overlap ratio (similar to ExpeL's approach)
                        if overlap > min(len(existing_words), len(new_words)) * 0.6:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        updated_insights.append(text)
        
        # Remove insights marked for deletion (in reverse order to maintain indices) - only if allow_delete
        if allow_delete:
            for idx in sorted(indices_to_delete, reverse=True):
                if idx < len(updated_insights):
                    del updated_insights[idx]
        
        # If no insights remain, use new insights as fallback
        if not updated_insights:
            updated_insights = list(new_insights)
        
        # Limit total insights to prevent unbounded growth
        max_insights = 50
        
        if len(updated_insights) > max_insights:
            # Keep most recent insights (could be improved with importance scoring like ExpeL)
            updated_insights = updated_insights[-max_insights:]
        
        return updated_insights
    
    def _update_state_abstracts(self, old_abstracts: List[Dict], new_abstracts: List[Dict], operations: List[Tuple[str, str]]) -> List[Dict]:
        """Apply operations to update state abstract list.
        
        This handles abstracts as dicts with 'state' and 'state_abstract' keys.
        """
        import re
        
        # Start with a copy of old abstracts
        updated_abstracts = list(old_abstracts)
        
        # Track indices to delete (process in reverse order)
        indices_to_delete = []
        
        # Process operations in order: REMOVE, EDIT, ADD
        operation_order = ['REMOVE', 'EDIT', 'ADD']
        
        for op_type in operation_order:
            for operation, text in operations:
                operation_type = operation.split()[0]
                
                if operation_type != op_type:
                    continue
                
                if operation_type == 'REMOVE':
                    # Extract abstract number
                    num_match = re.search(r'REMOVE (\d+)', operation)
                    if num_match:
                        idx = int(num_match.group(1)) - 1  # Convert to 0-based
                        if 0 <= idx < len(updated_abstracts):
                            indices_to_delete.append(idx)
                
                elif operation_type == 'EDIT':
                    # Extract abstract number
                    num_match = re.search(r'EDIT (\d+)', operation)
                    if num_match:
                        idx = int(num_match.group(1)) - 1  # Convert to 0-based
                        if 0 <= idx < len(updated_abstracts):
                            # Update the state_abstract string in the dict
                            updated_abstracts[idx]['state_abstract'] = text
                
                elif operation_type == 'ADD':
                    # Parse the state from the new abstract text if it follows the standard format
                    state_obj = None
                    if text.startswith("STATE: ") and "\nABSTRACT: " in text:
                        try:
                            state_part = text.split("\nABSTRACT: ")[0]
                            state_json_str = state_part.replace("STATE: ", "", 1)
                            state_obj = json.loads(state_json_str)
                        except:
                            pass
                    
                    # Create new abstract dict
                    new_abstract_dict = {
                        "state": state_obj,
                        "state_abstract": text
                    }
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing in updated_abstracts:
                        # Compare state objects if available
                        if state_obj and existing['state'] == state_obj:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        updated_abstracts.append(new_abstract_dict)
        
        # Remove abstracts marked for deletion (in reverse order to maintain indices)
        for idx in sorted(indices_to_delete, reverse=True):
            if idx < len(updated_abstracts):
                del updated_abstracts[idx]
        
        # If no abstracts remain, use new abstracts as fallback
        if not updated_abstracts:
            updated_abstracts = list(new_abstracts)
        
        # Limit total abstracts to prevent unbounded growth
        
        return updated_abstracts
    
    def _combine_simple_memory(self, old_memory: Dict, new_analysis: Dict, new_performance: Dict, new_games_count: int, new_abstracts: Dict) -> Dict:
        """Combine memory insights for simple format based on memory_merge_style."""
        # Use current generation performance (not weighted historical)
        total_games = old_memory["total_games"] + new_games_count
        combined_performance = new_performance  # Show current generation performance to agents
        
        # Get insights from old and new memory
        old_insights = old_memory["insights"]  # Already simple format
        new_insights = new_analysis["insights"]

        # Get analysis from old and new abstracts if available
        old_abstracts = old_memory["state_abstracts"]
        new_abstracts = new_abstracts["state_abstracts"]
        
        # Initialize merge logging fields
        merge_prompt = ""
        merge_response = ""
        operation_stats = ""
        
        # Handle state abstracts merging based on replay_merge_style
        replay_merge_prompt = ""
        replay_merge_response = ""
        replay_operation_stats = ""
        
        if self.replay_merge_style == "simple_add":
            # Simple add with state-based deduplication
            combined_abstracts = []
            seen_states = set()
            
            # Add new abstracts first (higher priority)
            for abstract in new_abstracts:
                state = abstract['state']
                state_key = json.dumps(state, sort_keys=True) if state else None
                if state_key not in seen_states:
                    combined_abstracts.append(abstract)
                    if state_key:
                        seen_states.add(state_key)
            
            # Add old abstracts that don't duplicate states
            for abstract in old_abstracts:
                state = abstract['state']
                state_key = json.dumps(state, sort_keys=True) if state else None
                if state_key not in seen_states:
                    combined_abstracts.append(abstract)
                    if state_key:
                        seen_states.add(state_key)
        elif self.replay_merge_style == "basic":
            # Use XML CRUD v2 operations for state abstracts
            combined_abstracts, replay_merge_prompt, replay_merge_response, replay_operation_stats = self._apply_state_abstract_operations_with_logging(old_abstracts, new_abstracts)
        else:
            raise ValueError(f"Invalid replay_merge_style: {self.replay_merge_style}. Must be one of: 'simple_add', 'basic'")
        
        # Apply merge behavior based on memory_merge_style (fail fast - no .get())
        merge_style = self.memory_merge_style
        
        if merge_style == "simple_add":
            # Add old memory to new memory - insights accumulate without limit
            combined_insights = list(new_insights) + list(old_insights)  # New first, then old
        
        elif merge_style == "no_merge":
            # Use only new insights, do not include old memory
            combined_insights = list(new_insights)
        
        elif merge_style == "basic":
            # XML CRUD v2 operations: XML-based with reliable parsing and strategic focus
            combined_insights, merge_prompt, merge_response, operation_stats = self._apply_memory_operations_with_logging(old_insights, new_insights, memory_merge_style=merge_style)
        
        else:
            raise ValueError(f"Invalid memory_merge_style: {merge_style}. Must be one of: 'simple_add', 'no_merge', 'basic'")
        
        # Build result dictionary
        result = {
            "total_games": total_games,
            "performance": combined_performance,
            "format": "simple",
            "insights": combined_insights,
            "merge_prompt": merge_prompt,  # LLM prompt used for merge (empty for non-LLM methods)
            "merge_response": merge_response,  # LLM response for merge (empty for non-LLM methods)
            "operation_stats": operation_stats,  # Operation counts (e.g., "ADD: 2; EDIT: 1")
            "state_abstracts": combined_abstracts,
            "replay_merge_prompt": replay_merge_prompt,  # LLM prompt used for state abstracts merge
            "replay_merge_response": replay_merge_response,  # LLM response for state abstracts merge
            "replay_operation_stats": replay_operation_stats  # Operation counts for state abstracts
        }
        
        return result


class MemoryEnhancedAgent:
    """Agent that uses trajectory-based insights for improved play."""
    
    def __init__(
        self,
        base_agent: ta.core.Agent,
        agent_name: str,
        memory_system: Optional[TrajectoryMemorySystem] = None,
        use_insights: bool = True,
        use_state_abstracts: bool = False,
        use_state_abstracts_match: bool = False,
        use_state_abstracts_reflex: bool = False,
        retrieval_enabled: bool = False,
        retrieval_model: str = "google/gemini-2.0-flash-001"
    ):
        """Initialize memory-enhanced agent.
        
        Args:
            base_agent: The base agent to enhance with memory
            agent_name: Name of the agent
            memory_system: The memory system to use (can be None if no memory is used)
            use_insights: Whether to load and use insights from memory (default: True)
            use_state_abstracts: Whether to load and use state abstracts from memory (default: False)
            use_state_abstracts_match: Whether to match current state with abstracts (default: False)
            use_state_abstracts_reflex: Whether to add reflexive warning about state abstract correctness (default: False)
            retrieval_enabled: Whether to use retrieval to select relevant insights
            retrieval_model: Model to use for retrieval
        """
        self.base_agent = base_agent
        self.agent_name = agent_name
        self.memory_system = memory_system
        self.last_memory_content = None  # Store last memory content used
        self.use_insights = use_insights
        self.use_state_abstracts = use_state_abstracts
        self.use_state_abstracts_match = use_state_abstracts_match
        self.use_state_abstracts_reflex = use_state_abstracts_reflex
        self.retrieval_enabled = retrieval_enabled
        
        # Assert that use_state_abstracts_match requires use_state_abstracts
        if self.use_state_abstracts_match:
            assert self.use_state_abstracts, "use_state_abstracts_match requires use_state_abstracts=True"
        
        # Assert that use_state_abstracts_reflex requires use_state_abstracts_match
        if self.use_state_abstracts_reflex:
            assert self.use_state_abstracts_match, "use_state_abstracts_reflex requires use_state_abstracts_match=True"
        
        # Initialize counters for state abstract matching
        self.match_call_count = 0
        self.match_success_count = 0
        
        # Create separate agent for retrieval if enabled
        if self.retrieval_enabled:
            try:
                from textarena.agents import OpenAIOpenrouterAgent
                self.retrieval_agent = OpenAIOpenrouterAgent(
                    model_name=retrieval_model,
                    system_prompt="You are a memory retrieval system. Extract only the most relevant insights for the current game situation."
                )
            except ImportError:
                self.retrieval_agent = ta.agents.OpenRouterAgent(
                    model_name=retrieval_model,
                    system_prompt="You are a memory retrieval system. Extract only the most relevant insights for the current game situation.",
                    track_tokens=False
                )
        else:
            self.retrieval_agent = None
    
    def _get_state_from_abstract_dict(self, abstract_dict: Dict) -> Optional[Dict]:
        """Get the state from an abstract dict.
        
        Args:
            abstract_dict: Dict containing 'state' and 'state_abstract' keys
            
        Returns:
            State dict or None if not found
        """
        return abstract_dict['state']
    
    
    def _find_matching_abstract(self, current_state: Dict, current_player_id: int, abstracts: List[Dict]) -> Optional[str]:
        """Find the abstract that matches the current game state AND current player.
        
        Args:
            current_state: Current game state as dict
            current_player_id: Current player ID (0 or 1)
            abstracts: List of state abstract dicts
            
        Returns:
            Matching abstract text or None
        """
        if not current_state:
            return None
        
        # Try to match with abstract states
        for abstract_dict in abstracts:
            abstract_state = self._get_state_from_abstract_dict(abstract_dict)
            abstract_player_id = abstract_dict['current_player_id']
            
            # Match both state AND current player
            if (abstract_state and abstract_state == current_state and 
                abstract_player_id is not None and abstract_player_id == current_player_id):
                # Extract just the ABSTRACT part from the full string
                full_abstract = abstract_dict['state_abstract']
                if "\nABSTRACT: " in full_abstract:
                    return full_abstract.split("\nABSTRACT: ", 1)[1]
        
        return None
    
    def __call__(self, observation: str, game_state: Optional[Dict] = None, player_id: Optional[int] = None) -> Tuple[str, str]:
        """Process observation with memory enhancement (hierarchical, flat, or simple).
        
        Args:
            observation: Game observation text
            game_state: Current game state dict (optional, used for state matching)
            player_id: Current player ID (optional, used for state matching)
        """
        # Reset last memory content
        self.last_memory_content = None
        
        # If neither insights nor state abstracts are enabled, just use base agent
        if not self.use_insights and not self.use_state_abstracts:
            return self.base_agent(observation), observation
        
        assert self.memory_system is not None, "Memory system is not provided"

        # Get shared memory data
        memory_data = self.memory_system.get_shared_memory()
        
        if memory_data:
            is_simple_format = memory_data["format"] == "simple"
            if is_simple_format:
                # Use simple memory format without structural labels
                return self._use_simple_memory(observation, memory_data, game_state, player_id)
            else:
                raise ValueError(f"Invalid memory format: {memory_data['format']}")
        else:
            return self.base_agent(observation), observation
        
    def retrieve(self, observation: str, memory_content: str) -> str:
        """Retrieve most relevant insights from memory for current observation."""
        if not self.retrieval_agent:
            raise ValueError("Retrieval agent not initialized. Set retrieval_enabled=True in constructor.")
        
        retrieval_prompt = f"""Given the current game observation and memory insights, identify the most relevant insights for this situation.
   
OBSERVATION:
{observation}
   
MEMORY INSIGHTS:
{memory_content}
   
Retrieve and return only the most relevant insights that would help with the current situation."""
        
        # Call dedicated retrieval agent to retrieve relevant insights
        retrieved_insights = self.retrieval_agent(retrieval_prompt)
        # print(f"DEBUG: Retrieved insights: {retrieved_insights}")
        # print(f"DEBUG: Memory content: {memory_content}")
        return retrieved_insights
    
    def _use_simple_memory(self, observation: str, memory_data: Dict, game_state: Optional[Dict] = None, player_id: Optional[int] = None) -> Tuple[str, str]:
        """Use simple memory format - list all insights without structural labels."""
        # Build memory content based on enabled features
        memory_sections = []
        
        # Add insights if enabled
        if self.use_insights:
            insights = memory_data['insights']
            serialized_insights = []
            for i, insight in enumerate(insights, 1):
                if insight and insight.strip():
                    serialized_insights.append(f"{i}. {insight}")
            
            if serialized_insights:
                insights_text = '\n'.join(serialized_insights)
                memory_sections.append(f"GAME INSIGHTS:\n\n{insights_text}")
        
        # Add state abstracts if enabled
        if self.use_state_abstracts:
            state_abstracts = memory_data['state_abstracts']
            
            if self.use_state_abstracts_match:
                # Find matching abstract for current state using direct state only
                if game_state and player_id is not None:
                    self.match_call_count += 1  # Increment call counter
                    matching_abstract = self._find_matching_abstract(game_state, player_id, state_abstracts)
                    # print(f"DEBUG: Matching abstract: {matching_abstract}")
                    if matching_abstract:
                        self.match_success_count += 1  # Increment success counter
                        if self.use_state_abstracts_reflex:
                            # Add reflexive warning about abstract correctness
                            reflex_warning = ("NOTE: The following state abstract may be correct or wrong. "
                                            "You need to check and do deep analysis to verify whether it's correct "
                                            "and can apply to the current state to guide your next action.\n\n")
                            memory_sections.append(f"MATCHING STATE ABSTRACT:\n\n{reflex_warning}{matching_abstract}")
                        else:
                            memory_sections.append(f"MATCHING STATE ABSTRACT:\n\n{matching_abstract}")
                    else:
                        # print(f"DEBUG: No matching abstract found for game state: {game_state}")
                        pass
                else:
                    raise ValueError("Game state and player ID are required for state abstract matching")
            else:
                # Original behavior: list all abstracts
                state_abstracts_list = []
                for i, abstract_dict in enumerate(state_abstracts, 1):
                    # Extract the full state_abstract string for display (direct access)
                    full_abstract = abstract_dict['state_abstract']
                    if full_abstract and full_abstract.strip():
                        state_abstracts_list.append(f"{i}. {full_abstract}")
                
                if state_abstracts_list:
                    state_abstracts_text = '\n'.join(state_abstracts_list)
                    memory_sections.append(f"RELEVANT TRAJECTORY ABSTRACTS:\n\n{state_abstracts_text}")
        
        # If no memory content is enabled, just use base agent
        if not memory_sections:
            return self.base_agent(observation), observation
        
        # Combine all memory sections
        memory_content = '\n\n'.join(memory_sections)
        
        # Store the memory content
        self.last_memory_content = memory_content
        
        if self.retrieval_enabled:
            # Use retrieval to get most relevant insights
            retrieved_content = self.retrieve(observation, memory_content)
            
            # Create enhancement with retrieved insights
            enhancement = f"""GAME INSIGHTS:

{retrieved_content}

{observation}"""
        else:
            # Original behavior: concatenate all memory content
            enhancement = f"""
            Given the current game observation and memory insights, first retrieve the most relevant insights, then choose the best action to take.
            Game Insights: {memory_content}
            Observation: {observation}
            Given the current game observation and memory insights, first retrieve the most relevant insights, then choose the best action to take.            
            """

        return self.base_agent(enhancement), enhancement
    