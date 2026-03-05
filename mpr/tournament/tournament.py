from dataclasses import dataclass
import logging
from pathlib import Path
from collections import defaultdict

from mpr.memory.trajectory_memory_system import MemoryEnhancedAgent
from mpr.tournament.agent_pool import AgentPool, AgentPerformance
from mpr.replaybuffer.replaybuffer import ReplayBuffer
from mpr.evaluation.simple_memory_eval import run_tournament
from mpr.utils.output_manager import OutputManager
from mpr.prompts.prompt_evolution_engine import PromptCandidate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TournamentResult:
    generation_id: int
    games_played: int
    stats: dict[str, any] = None
    agent_performance: dict[str, AgentPerformance] = None
    trajectories: list[dict] = None
    output_dir: Path | None = None


def init_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class Tournament:
    def __init__(
        self,
        baseline_model: str,
        max_concurrent: int,
        temperature: float,
        output_manager: OutputManager,
        model_name: str = None,
        memory_system=None,
        use_replay: bool = False,
        use_state_abstracts_match: bool = False,
        use_state_abstracts_reflex: bool = False,
        memory_retrieval_enabled: bool = False,
        prompt_debug: bool = False,
        skip_baseline_eval: bool = False,
        eval_model_list: list[str] | None = None,
        generalize_model_list: list[str] | None = None,
        env_id: str = None,
        eval_model_evolution: dict[str, dict[str, list]] | None = None,
        generalize_model_evolution: dict[str, dict[str, list]] | None = None,
    ):
        self.baseline_model = baseline_model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.output_manager = output_manager
        self.model_name = model_name or baseline_model
        self.memory_system = memory_system
        self.use_replay = use_replay
        self.use_state_abstracts_match = use_state_abstracts_match
        self.use_state_abstracts_reflex = use_state_abstracts_reflex
        self.memory_retrieval_enabled = memory_retrieval_enabled
        self.prompt_debug = prompt_debug
        self.skip_baseline_eval = skip_baseline_eval
        self.eval_model_list = eval_model_list or []
        self.generalize_model_list = generalize_model_list
        self.env_id = env_id

        # Evolution tracking dictionaries
        self.eval_model_evolution = eval_model_evolution
        self.generalize_model_evolution = generalize_model_evolution

    def run_generation(
        self,
        generation_id: int,
        env_id: str,
        agent_pool: AgentPool,
        rounds: int,
        phase: str,
        folder_name: str,
        replay_buffer: ReplayBuffer | None = None,
        schedule_type: str = "vs_baseline",
        baseline_model: str | None = None,
        track_tokens: bool = False,
        **tournament_kwargs,
    ) -> TournamentResult:
        """Run a tournament generation.
        
        Args:
            generation_id: Current generation number
            env_id: Environment ID to run on
            agent_pool: Pool of agents to compete
            rounds: Number of rounds
            phase: Phase name for logging
            folder_name: Output folder name
            replay_buffer: Optional replay buffer
            schedule_type: Tournament schedule type ("vs_baseline" or "vs_best")
            baseline_model: Override baseline model (defaults to self.baseline_model)
            track_tokens: Whether to track tokens during the tournament
            **tournament_kwargs: Additional args passed to run_tournament
        """
        logger.info(f"Gen {generation_id}: Running tournament with {rounds} rounds")

        output_dir = init_path(
            self.output_manager.project_root / "temp" / folder_name
        )

        tournament_args = dict(
            agent_pool=agent_pool,
            env_id=env_id,
            num_players_per_game=2,
            num_rounds=rounds,
            max_concurrent=self.max_concurrent,
            output_dir=str(output_dir),
            memory_agents=[],
            generation=generation_id,
            phase=phase,
            schedule_type=schedule_type,
            baseline_model=baseline_model or self.baseline_model,
            temperature=self.temperature,
            replay_buffer=replay_buffer,
            track_tokens=track_tokens,
            **tournament_kwargs,
        )

        stats, results = run_tournament(**tournament_args)

        logger.info(f"Tournament complete for generation {generation_id}")

        self.output_manager.organize_tournament_files(
            output_dir,
            generation_id,
            evaluation_phase=False,
        )

        return TournamentResult(
            generation_id=generation_id,
            games_played=stats.get("games_played", 0),
            stats=stats,
            agent_performance={
                agent_id: agent_pool.get_agent_performance(agent_id)
                for agent_id in agent_pool.get_all_agent_ids()
            },
            trajectories=[r.trajectory for r in results],
            output_dir=output_dir,
        )

    def run_evaluation(
        self,
        eval_env_ids: list[str],
        generation_id: int,
        rounds: int,
        best_prompt_candidate: PromptCandidate,
    ) -> tuple[dict, dict, dict, AgentPerformance, AgentPerformance, dict, dict]:
        """Run evaluation tournament with extended evaluation against eval_model_list.
        
        Returns:
            tuple of (eval_model_evolution, generalize_model_evolution, empty_stats, 
                      empty_perf, empty_perf, env_stats, env_performance)
        """
        logger.info(f"Gen {generation_id}: Running evaluation tournaments")

        # Check if we're using generalize_model_list mode
        if self.generalize_model_list:
            return self._run_generalize_evaluation(generation_id, rounds, best_prompt_candidate)

        env_stats, env_performance = {}, {}

        for eval_env_id in eval_env_ids:
            logger.info(f"\n{'#'*60}")
            logger.info(f"Evaluating on environment: {eval_env_id}")
            logger.info(f"{'#'*60}")

            all_stats = {}
            all_performance = {}

            # Step 1: For generation 0, run eval_model_list vs baseline (unless skipped)
            if generation_id == 0 and self.eval_model_list and not self.skip_baseline_eval:
                baseline_stats = self._run_baseline_evaluation(eval_env_id, generation_id, rounds)
                all_stats["vs_baseline"] = baseline_stats

            elif generation_id == 0 and self.skip_baseline_eval:
                logger.info(f"Skipping baseline evaluation for generation {generation_id} on {eval_env_id} (skip_baseline_eval=True)")
                for eval_model in self.eval_model_list:
                    self.eval_model_evolution[eval_env_id][eval_model].append({
                        'generation': generation_id,
                        'opponent': 'baseline',
                        'win_rate': 0.0,
                        'draw_rate': 0.0
                    })
                logger.info("Set baseline performance to 0.0 for all eval models to handle dependencies")

            # Step 2: For all generations, run eval_model_list vs best_candidate
            if self.eval_model_list:
                best_stats, best_perf = self._run_best_candidate_evaluation(eval_env_id, generation_id, rounds, best_prompt_candidate)
                all_stats["vs_best"] = best_stats
                all_performance["best_candidate"] = best_perf

            env_stats[eval_env_id] = all_stats
            env_performance[eval_env_id] = all_performance

        # Return empty stats instead of None to avoid downstream issues (for backward compatibility)
        empty_stats = {}
        empty_perf = AgentPerformance(agent_id="dummy")
        
        return self.eval_model_evolution, self.generalize_model_evolution, empty_stats, empty_perf, empty_perf, env_stats, env_performance

    def _run_baseline_evaluation(
        self,
        eval_env_id: str,
        generation_id: int,
        rounds: int,
    ) -> dict:
        """Run evaluation of eval_model_list vs baseline using run_generation.
        
        Returns:
            Tournament stats dict
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Step 1: eval_model_list vs baseline on {eval_env_id}")
        logger.info(f"{'='*50}")

        # Create evaluation pool
        eval_pool = AgentPool(prompt_debug=self.prompt_debug)
        eval_pool.create_agents_from_models(
            model_names=self.eval_model_list,
            templates=["gemini-boxed"] * len(self.eval_model_list),
            evolved_prompts=[None] * len(self.eval_model_list),
            temperature=self.temperature,
        )
        
        # Use run_generation instead of calling run_tournament directly
        result = self.run_generation(
            generation_id=generation_id,
            env_id=eval_env_id,
            agent_pool=eval_pool,
            rounds=rounds,
            phase="vs_baseline",
            folder_name=f"eval_gen{generation_id}_env_{eval_env_id.replace('/', '_')}_vs_baseline",
            schedule_type="vs_baseline",
        )

        # Print results
        logger.info(f"\nEvaluation vs Baseline Results on {eval_env_id}:")
        logger.info("-" * 80)

        agent_ids = eval_pool.get_all_agent_ids()

        for i, eval_model in enumerate(self.eval_model_list):
            if i < len(agent_ids):
                agent_id = agent_ids[i]
                perf = eval_pool.get_agent_performance(agent_id)
                if perf:
                    win_rate, loss_rate, draw_rate = perf.rates_vs_opponents(perspective="baseline")
                    logger.info(f"Baseline vs {eval_model:30s}: W:{win_rate:.1%} L:{loss_rate:.1%} D:{draw_rate:.1%}")

                    # Track performance vs baseline (only for generation 0)
                    self.eval_model_evolution[eval_env_id][eval_model].append({
                        'generation': generation_id,
                        'opponent': 'baseline',
                        'win_rate': loss_rate,  # Eval model's win rate vs baseline
                        'draw_rate': draw_rate
                    })

        logger.info(f"{'Baseline (' + self.baseline_model + ')':40s}: Performance not tracked in vs_baseline mode")
        logger.info("-" * 80)

        return result.stats

    def _run_best_candidate_evaluation(
        self,
        eval_env_id: str,
        generation_id: int,
        rounds: int,
        best_prompt_candidate: PromptCandidate,
    ) -> tuple[dict, AgentPerformance]:
        """Run evaluation of eval_model_list vs best candidate using run_generation.
        
        Returns:
            tuple of (stats dict, best_candidate AgentPerformance)
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Step 2: eval_model_list vs best_candidate on {eval_env_id}")
        logger.info(f"{'='*50}")

        # Create new evaluation pool
        eval_pool = AgentPool(prompt_debug=self.prompt_debug)

        # Add best candidate first
        best_candidate_id = f"best_candidate_{best_prompt_candidate.id}"
        eval_pool.add_agent(
            agent_id=best_candidate_id,
            model_name=self.model_name,
            template_name="mpr-evolved",
            evolved_prompt=best_prompt_candidate.prompt,
            temperature=self.temperature,
        )

        # Wrap with MemoryEnhancedAgent
        base_agent = eval_pool.agents[best_candidate_id]
        memory_agent = MemoryEnhancedAgent(
            base_agent=base_agent,
            agent_name=best_candidate_id,
            memory_system=self.memory_system,
            use_insights=False,
            use_state_abstracts=False,
            use_state_abstracts_match=self.use_state_abstracts_match,
            use_state_abstracts_reflex=self.use_state_abstracts_reflex,
            retrieval_enabled=self.memory_retrieval_enabled,
            retrieval_model=self.model_name,
        )
        eval_pool.agents[best_candidate_id] = memory_agent

        # Add all eval models
        eval_model_to_agent_id = {}
        for i, eval_model in enumerate(self.eval_model_list):
            eval_agent_id = f"eval_model_{i}_{eval_model.split('/')[-1]}"
            eval_pool.add_agent(
                agent_id=eval_agent_id,
                model_name=eval_model,
                template_name="gemini-boxed",
                evolved_prompt=None,
                temperature=self.temperature,
            )
            eval_model_to_agent_id[eval_model] = eval_agent_id

        agent_ids = eval_pool.get_all_agent_ids()
        logger.info(f"Agent pool order for vs_best: {agent_ids}")
        logger.info(f"First agent (best candidate): {agent_ids[0]}")
        logger.info(f"Eval model to agent ID mapping: {eval_model_to_agent_id}")

        # Use run_generation with vs_best schedule
        result = self.run_generation(
            generation_id=generation_id,
            env_id=eval_env_id,
            agent_pool=eval_pool,
            rounds=rounds,
            phase="vs_best",
            folder_name=f"eval_gen{generation_id}_env_{eval_env_id.replace('/', '_')}_vs_best",
            schedule_type="vs_best",
        )

        # Print state abstract match statistics if enabled
        if self.use_state_abstracts_match and self.use_replay:
            best_agent = eval_pool.agents.get(best_candidate_id)
            if hasattr(best_agent, 'match_call_count') and best_agent.match_call_count > 0:
                match_rate = best_agent.match_success_count / best_agent.match_call_count * 100
                logger.info(f"\nState Abstract Matching Statistics for {eval_env_id}:")
                logger.info(f"Total match attempts: {best_agent.match_call_count}")
                logger.info(f"Successful matches: {best_agent.match_success_count}")
                logger.info(f"Match rate: {match_rate:.1f}%")
                logger.info("-" * 80)

        # Print results
        logger.info(f"\nEvaluation vs Best Candidate Results on {eval_env_id}:")
        logger.info(f"Best candidate: {best_prompt_candidate.id} (model: {self.model_name})")
        logger.info("-" * 80)

        best_perf = eval_pool.get_agent_performance(best_candidate_id)

        for eval_model in self.eval_model_list:
            agent_id = eval_model_to_agent_id.get(eval_model)
            perf = eval_pool.get_agent_performance(agent_id) if agent_id else None
            if perf:
                win_rate, loss_rate, draw_rate = perf.rates_vs_opponents(perspective="baseline")
                logger.info(f"Best Candidate vs {eval_model}: W:{win_rate:.1%} L:{loss_rate:.1%} D:{draw_rate:.1%}")

                # Track performance vs baseline (only for generation 0)
                self.eval_model_evolution[eval_env_id][eval_model].append({
                    'generation': generation_id,
                    'opponent': 'best_candidate',
                    'win_rate': loss_rate,  # Eval model's win rate vs baseline
                    'draw_rate': draw_rate
                })

        # Print best candidate overall summary
        if best_perf:
            replay_status = "(using replay)" if self.use_replay else ""
            _, loss_rate, draw_rate = best_perf.rates_vs_opponents(perspective="baseline")

            logger.info("-" * 80)
            logger.info(
                f"Best Candidate ({self.model_name}) {replay_status} Overall on {eval_env_id}: "
                f"W:{best_perf.win_rate():.1%} L:{loss_rate:.1%} D:{draw_rate:.1%} "
                f"(TrueSkill: {best_perf.trueskill_rating.mu:.2f})"
            )

            # Calculate average best candidate win rate
            total_best_winrate = 0.0
            count = 0
            for eval_model in self.eval_model_list:
                if self.eval_model_evolution[eval_env_id][eval_model]:
                    last_entry = self.eval_model_evolution[eval_env_id][eval_model][-1]
                    if last_entry['opponent'] == 'best_candidate':
                        dr = last_entry.get('draw_rate', 0.0)
                        total_best_winrate += 1.0 - last_entry['win_rate'] - dr
                        count += 1

            if count > 0:
                logger.info(
                    f"Best Candidate Average Win Rate vs Eval Models on {eval_env_id}: "
                    f"{total_best_winrate / count:.1%}"
                )

        logger.info("-" * 80)

        return result.stats, best_perf

    def _run_generalize_evaluation(
        self,
        generation_id: int,
        rounds: int,
        best_prompt_candidate: PromptCandidate,
    ) -> tuple[dict, dict, dict, AgentPerformance, AgentPerformance, dict, dict]:
        """Run evaluation tournament for generalize_model_list mode.
        
        Returns:
            tuple of (eval_model_evolution, generalize_model_evolution, empty_stats,
                      empty_perf, empty_perf, model_stats, model_performance)
        """
        logger.info(f"Gen {generation_id}: Running generalization evaluation")
        logger.info(f"Testing {len(self.generalize_model_list)} models with evolved prompt")

        eval_env_id = self.env_id
        model_stats = {}
        model_performance = {}

        for generalize_model in self.generalize_model_list:
            logger.info(f"\n{'#'*60}")
            logger.info(f"Testing generalization with model: {generalize_model}")
            logger.info(f"{'#'*60}")

            all_stats = {}
            all_performance = {}

            # Step 1: For generation 0, run baseline evaluation
            if generation_id == 0 and not self.skip_baseline_eval:
                baseline_stats = self._run_generalize_baseline_evaluation(
                    eval_env_id, generation_id, rounds, generalize_model
                )
                all_stats["vs_baseline"] = baseline_stats

            # Step 2: Run eval_model_list vs generalize_model with evolved prompt
            evolved_stats, gen_perf, model_metrics = self._run_generalize_evolved_evaluation(
                eval_env_id, generation_id, rounds, best_prompt_candidate, generalize_model
            )
            all_stats["vs_evolved"] = evolved_stats
            all_performance["generalize_model"] = gen_perf
            all_performance["vs_eval_models"] = model_metrics

            model_stats[generalize_model] = all_stats
            model_performance[generalize_model] = all_performance

        # Return empty stats instead of None to avoid downstream issues (for backward compatibility)
        empty_stats = {}
        empty_perf = AgentPerformance(agent_id="dummy")
        
        return self.eval_model_evolution, self.generalize_model_evolution, empty_stats, empty_perf, empty_perf, model_stats, model_performance

    def _run_generalize_baseline_evaluation(
        self,
        eval_env_id: str,
        generation_id: int,
        rounds: int,
        generalize_model: str,
    ) -> dict:
        """Run baseline evaluation for a generalize model using run_generation."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Baseline: {generalize_model} vs eval_model_list")
        logger.info(f"{'='*50}")

        eval_pool = AgentPool(prompt_debug=self.prompt_debug)
        eval_pool.create_agents_from_models(
            model_names=self.eval_model_list,
            templates=["gemini-boxed"] * len(self.eval_model_list),
            evolved_prompts=[None] * len(self.eval_model_list),
            temperature=self.temperature,
        )

        # Use run_generation with generalize_model as baseline
        result = self.run_generation(
            generation_id=generation_id,
            env_id=eval_env_id,
            agent_pool=eval_pool,
            rounds=rounds,
            phase="generalize_baseline",
            folder_name=f"eval_gen{generation_id}_generalize_{generalize_model.replace('/', '_')}_baseline",
            schedule_type="vs_baseline",
            baseline_model=generalize_model,  # Override baseline
        )

        # Print results
        logger.info(f"\nBaseline Results - {generalize_model} vs eval_model_list:")
        logger.info("-" * 80)

        agent_ids = eval_pool.get_all_agent_ids()
        total_games, total_wins, total_losses, total_draws = 0, 0, 0, 0

        for i, eval_model in enumerate(self.eval_model_list):
            if i < len(agent_ids):
                agent_id = agent_ids[i]
                perf = eval_pool.get_agent_performance(agent_id)
                if perf:
                    win_rate, loss_rate, draw_rate = perf.rates_vs_opponents(perspective="baseline")
                    logger.info(f"{generalize_model} vs {eval_model}: W:{win_rate:.1%} L:{loss_rate:.1%} D:{draw_rate:.1%}")

                    self.generalize_model_evolution[generalize_model][eval_model].append({
                        'generation': generation_id,
                        'win_rate': win_rate,
                        'draw_rate': draw_rate,
                        'opponent_type': 'baseline'
                    })

                    total_games += perf.games_played
                    total_wins += perf.losses
                    total_losses += perf.wins
                    total_draws += perf.draws

        if total_games > 0:
            logger.info("-" * 80)
            logger.info(f"{generalize_model} Overall Baseline Performance:")
            logger.info(f"  Games: {total_games}, W:{total_wins/total_games:.1%} L:{total_losses/total_games:.1%} D:{total_draws/total_games:.1%}")

        logger.info("-" * 80)

        return result.stats

    def _run_generalize_evolved_evaluation(
        self,
        eval_env_id: str,
        generation_id: int,
        rounds: int,
        best_prompt_candidate: PromptCandidate,
        generalize_model: str,
    ) -> tuple[dict, AgentPerformance, dict]:
        """Run evaluation with evolved prompt for a generalize model using run_generation."""
        logger.info(f"\n{'='*50}")
        logger.info(f"With evolved prompt: {generalize_model} vs eval_model_list")
        logger.info(f"{'='*50}")

        eval_pool = AgentPool(prompt_debug=self.prompt_debug)

        # Add generalize model with evolved prompt
        gen_agent_id = f"generalize_{generalize_model.split('/')[-1]}_{best_prompt_candidate.id}"
        eval_pool.add_agent(
            agent_id=gen_agent_id,
            model_name=generalize_model,
            template_name="mpr-evolved",
            evolved_prompt=best_prompt_candidate.prompt,
            temperature=self.temperature,
        )

        # Wrap with MemoryEnhancedAgent
        base_agent = eval_pool.agents[gen_agent_id]
        memory_agent = MemoryEnhancedAgent(
            base_agent=base_agent,
            agent_name=gen_agent_id,
            memory_system=self.memory_system,
            use_insights=False,
            use_state_abstracts=self.use_replay,
            use_state_abstracts_match=self.use_state_abstracts_match,
            use_state_abstracts_reflex=self.use_state_abstracts_reflex,
            retrieval_enabled=self.memory_retrieval_enabled,
            retrieval_model=generalize_model,
        )
        eval_pool.agents[gen_agent_id] = memory_agent

        # Add eval models
        eval_model_to_agent_id = {}
        for i, eval_model in enumerate(self.eval_model_list):
            eval_agent_id = f"eval_model_{i}_{eval_model.split('/')[-1]}"
            eval_pool.add_agent(
                agent_id=eval_agent_id,
                model_name=eval_model,
                template_name="gemini-boxed",
                evolved_prompt=None,
                temperature=self.temperature,
            )
            eval_model_to_agent_id[eval_model] = eval_agent_id

        # Use run_generation with vs_best schedule
        result = self.run_generation(
            generation_id=generation_id,
            env_id=eval_env_id,
            agent_pool=eval_pool,
            rounds=rounds,
            phase="generalize_evolved",
            folder_name=f"eval_gen{generation_id}_generalize_{generalize_model.replace('/', '_')}_evolved",
            schedule_type="vs_best",
        )

        # Print results
        logger.info(f"\nEvolved Prompt Results - {generalize_model} vs eval_model_list:")
        logger.info(f"Using evolved prompt from: {best_prompt_candidate.id}")
        logger.info("-" * 80)

        gen_perf = eval_pool.get_agent_performance(gen_agent_id)
        model_metrics = {}

        for eval_model in self.eval_model_list:
            agent_id = eval_model_to_agent_id.get(eval_model)
            if agent_id:
                perf = eval_pool.get_agent_performance(agent_id)
                if perf:
                    win_rate, loss_rate, draw_rate = perf.rates_vs_opponents(perspective="baseline")
                    logger.info(f"{generalize_model} vs {eval_model}: W:{win_rate:.1%} L:{loss_rate:.1%} D:{draw_rate:.1%}")

                    model_metrics[eval_model] = {
                        'win_rate': win_rate,
                        'loss_rate': loss_rate,
                        'draw_rate': draw_rate,
                    }

                    self.generalize_model_evolution[generalize_model][eval_model].append({
                        'generation': generation_id,
                        'win_rate': win_rate,
                        'draw_rate': draw_rate,
                        'opponent_type': 'evolved'
                    })

        # Print overall
        if gen_perf and gen_perf.games_played > 0:
            _, loss_rate, draw_rate = gen_perf.rates_vs_opponents(perspective="baseline")

            logger.info("-" * 80)
            logger.info(
                f"{generalize_model} Overall: "
                f"W:{gen_perf.win_rate():.1%} L:{loss_rate:.1%} D:{draw_rate:.1%} "
                f"(TrueSkill: {gen_perf.trueskill_rating.mu:.2f})"
            )

            if model_metrics:
                avg_win_rate = sum(m['win_rate'] for m in model_metrics.values()) / len(model_metrics)
                logger.info(f"Average Win Rate vs Eval Models: {avg_win_rate:.1%}")

        elif model_metrics:
            total_win = sum(m['win_rate'] for m in model_metrics.values()) / len(model_metrics)
            total_loss = sum(m['loss_rate'] for m in model_metrics.values()) / len(model_metrics)
            total_draw = sum(m['draw_rate'] for m in model_metrics.values()) / len(model_metrics)

            logger.info("-" * 80)
            logger.info(
                f"{generalize_model} Overall (from matchups): "
                f"W:{total_win:.1%} L:{total_loss:.1%} D:{total_draw:.1%}"
            )

        logger.info("-" * 80)

        return result.stats, gen_perf, model_metrics
