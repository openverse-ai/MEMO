#!/usr/bin/env python3
"""
Self-Play Prompt Evolution System
Evolves prompts through tournament-based selection and performance feedback.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict
import wandb
from dotenv import load_dotenv
import hashlib
from textarena.agents.basic_agents import get_total_tokens_used, get_total_self_play_tokens_used, get_total_optimization_tokens_used


# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

wandb.login(key=os.getenv("WANDB_API_KEY"))
from mpr.tournament.agent_pool import AgentPool
from mpr.prompts.prompt_evolution_engine import PromptEvolutionEngine
from mpr.memory.trajectory_memory_system import TrajectoryMemorySystem, CompressedGame
from mpr.memory.prompts import BASIC_ABSTRACT_GEN_PROMPT
from mpr.replaybuffer.replaybuffer import ReplayBuffer


from mpr.tournament.tournament import Tournament
from mpr.utils.output_manager import OutputManager
from mpr.utils.evolution_reporter import (
    print_eval_model_evolution,
    print_generalize_model_evolution,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SelfPlayPromptEvolution:
    def __init__(
        self,
        model_name,
        baseline_model,
        env_id="TicTacToe-v0",
        population_size=10,
        analyzer_model="google/gemini-2.0-flash-001",
        script_name=None,
        trajectories_path=None,
        max_concurrent=50,
        # Evolution strategy ratios
        keep_ratio=0.3,
        random_ratio=0.2,
        memory_guided_ratio=0.0,
        trajectory_ratio=0.3,
        crossover_ratio=0.2,
        # Verification settings
        use_importance_ranking=False,
        # Memory system settings
        memory_merge_style="basic",
        max_games_in_prompt=1,
        game_sequence_style="raw_action_only",
        # Reflection settings
        max_games_per_agent_reflection=None,
        # Fitness method
        fitness_method="trueskill",
        # Model lists
        eval_model_list=None,
        generalize_model_list=None,
        # Replay settings
        use_replay=False,
        buffer_capacity=100000,
        alpha=0.6,
        beta=0.4,
        replay_max_steps=None,
        # Sampling
        temperature=0.0,
        replay_topk=1,
        replay_merge_style="basic",
        abstract_gen_style="basic",
        skip_baseline_eval=False,
        prompt_debug=False,
        insight_sampling_mode="sample",
    ):
        # -------------------------
        # Basic assignments
        # -------------------------
        self.model_name = model_name
        self.baseline_model = baseline_model
        self.env_id = env_id
        self.population_size = population_size
        self.analyzer_model = analyzer_model
        self.trajectories_path = trajectories_path
        self.max_concurrent = max_concurrent

        self.keep_ratio = keep_ratio
        self.random_ratio = random_ratio
        self.memory_guided_ratio = memory_guided_ratio
        self.trajectory_ratio = trajectory_ratio
        self.crossover_ratio = crossover_ratio

        self.max_games_in_prompt = max_games_in_prompt
        self.game_sequence_style = game_sequence_style
        self.memory_merge_style = memory_merge_style

        self.max_games_per_agent_reflection = max_games_per_agent_reflection
        self.fitness_method = fitness_method

        self.eval_model_list = eval_model_list or []
        self.generalize_model_list = generalize_model_list

        self.use_replay = use_replay
        self.replay_max_steps = replay_max_steps
        self.replay_topk = replay_topk
        self.replay_merge_style = replay_merge_style
        self.abstract_gen_style = abstract_gen_style

        self.skip_baseline_eval = skip_baseline_eval
        self.prompt_debug = prompt_debug
        self.insight_sampling_mode = insight_sampling_mode
        self.temperature = temperature

        # Memory always enabled for guided generation
        self.memory_enable = True

        self.global_seen_states = set()
        self.global_seen_trajectories = set()

        # Track eval model performance evolution across generations
        # Design decision: We ALWAYS store the eval model's win rate for consistency
        # and only invert it at display time to show baseline/best candidate's perspective
        self.eval_model_evolution = {}
        eval_envs = [self.env_id]
        for env_id in eval_envs:
            self.eval_model_evolution[env_id] = {}
            if eval_model_list:
                for model in eval_model_list:
                    self.eval_model_evolution[env_id][model] = []
        
        # Track generalize model evolution across generations
        # Structure: {generalize_model: {eval_model: [{generation, win_rate, draw_rate, opponent_type}]}}
        self.generalize_model_evolution = {}
        if generalize_model_list:
            for gen_model in generalize_model_list:
                self.generalize_model_evolution[gen_model] = {}
                if eval_model_list:
                    for eval_model in eval_model_list:
                        self.generalize_model_evolution[gen_model][eval_model] = []


        # validations
        ratio_sum = keep_ratio + random_ratio + memory_guided_ratio + trajectory_ratio + crossover_ratio
        if abs(ratio_sum - 1.0) > 1e-3:
            raise ValueError(
                "Evolution strategy ratios must sum to 1.0 "
                f"(got {ratio_sum:.3f}; "
                f"keep={keep_ratio}, random={random_ratio}, memory_guided={memory_guided_ratio}, "
                f"trajectory={trajectory_ratio}, crossover={crossover_ratio})"
            )
        if population_size > 3 and max_games_per_agent_reflection is None:
            raise ValueError(
                "max_games_per_agent_reflection must be set when population_size > 3 "
                f"(population_size={population_size})"
            )
        if self.generalize_model_list and not self.eval_model_list:
            raise ValueError(
                "generalize_model_list requires eval_model_list to be set"
            )
        # Project directories
        project_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{script_name}"
        self.output_manager = OutputManager.init(project_name=project_name)

        # tournamnet
        self.tournament = Tournament(
            baseline_model=self.baseline_model, max_concurrent=self.max_concurrent,
            temperature=self.temperature, output_manager=self.output_manager, prompt_debug=self.prompt_debug,
            skip_baseline_eval=self.skip_baseline_eval,
            eval_model_list=self.eval_model_list, generalize_model_list=self.generalize_model_list,
            eval_model_evolution=self.eval_model_evolution, generalize_model_evolution=self.generalize_model_evolution
        )

        # replay buffer
        self.replay_buffer = (
            ReplayBuffer(
                buffer_capacity=buffer_capacity,
                alpha=alpha,
                beta=beta,
                max_steps=replay_max_steps,
            )
            if self.use_replay
            else None
        )

        # Memory system
        self.memory_system = TrajectoryMemorySystem(
            memory_merge_style=memory_merge_style,
            insights_dir=str(self.output_manager.memory_dir),
            analyzer_model=analyzer_model,
            use_state_memory=False,
            prompt_style="simple",
            max_games_in_prompt=max_games_in_prompt,
            game_sequence_style=game_sequence_style,
            replay_merge_style=replay_merge_style,
            prompt_debug=prompt_debug,
        )

        # Prompt evolution engine
        self.prompt_engine = PromptEvolutionEngine(
            population_size=population_size,
            analyzer_model=analyzer_model,
            env_id=env_id,
            output_dir=str(self.output_manager.prompts_dir / "evolution"),
            keep_ratio=keep_ratio,
            random_ratio=random_ratio,
            memory_guided_ratio=memory_guided_ratio,
            trajectory_ratio=trajectory_ratio,
            crossover_ratio=crossover_ratio,
            use_importance_ranking=use_importance_ranking,
            fitness_method=fitness_method,
            temperature=self.temperature,
            memory_system=self.memory_system,
            prompt_debug=prompt_debug,
            insight_sampling_mode=self.insight_sampling_mode,
        )

        # wandb
        self._init_wandb(name=project_name)

        # logging
        logger.info("ReplayBuffer %s for evolution", "ENABLED" if self.use_replay else "DISABLED")
        logger.info("Prompt Evolution System initialized")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Baseline: {self.baseline_model}")
        logger.info(f"  Environment: {self.env_id}")
        logger.info(f"  Population Size: {self.population_size}")
        logger.info(f"  Fitness Method: {self.fitness_method}")
        logger.info(f"  Replay Enabled: {self.use_replay}")
        logger.info(f"  Eval Models: {self.eval_model_list or 'None'}")
        logger.info(f"  Generalize Models: {self.generalize_model_list or 'None'}")

    def _init_wandb(self, name: str, project: str = "prompt-evolution"):
        wandb.init(
            entity="i2r-ali",
            project=project,
            name=name,
            config={
                "model_name": self.model_name,
                "baseline_model": self.baseline_model,
                "env_id": self.env_id,
                "population_size": self.population_size,
                "keep_ratio": self.keep_ratio,
                "random_ratio": self.random_ratio,
                "memory_guided_ratio": self.memory_guided_ratio,
                "trajectory_ratio": self.trajectory_ratio,
                "crossover_ratio": self.crossover_ratio,
                "analyzer_model": self.analyzer_model,
                "trajectories_path": self.output_manager.trajectories_dir,
                "project_root": str(self.output_manager.project_root),
                "max_concurrent": self.max_concurrent,
                "memory_merge_style": self.memory_merge_style,
                "max_games_in_prompt": self.max_games_in_prompt,
                "game_sequence_style": self.game_sequence_style,
                "memory_enable": self.memory_enable,
                "max_games_per_agent_reflection": self.max_games_per_agent_reflection,
                "fitness_method": self.fitness_method,
                "eval_model_list": self.eval_model_list,
                "generalize_model_list": self.generalize_model_list,
                "temperature": self.temperature,
                "replay_topk": self.replay_topk,
                "replay_merge_style": self.replay_merge_style,
                "abstract_gen_style": self.abstract_gen_style,
                "skip_baseline_eval": self.skip_baseline_eval,
                "replay_max_steps": self.replay_max_steps,
                "prompt_debug": self.prompt_debug,
                "insight_sampling_mode": self.insight_sampling_mode,
            },
        )
    
    def run_evolution(self, base_prompt, num_generations, tournament_rounds, eval_rounds):
        """Run complete evolution process."""
        logger.info(f"Starting {num_generations} generation evolution")
        logger.info(f"Tournament rounds: {tournament_rounds}, Eval rounds: {eval_rounds}")
        
        results = []
        
        # Generation 0: Create initial population
        logger.info("\n" + "="*50)
        logger.info("GENERATION 0: Initial Population")
        logger.info("="*50)
        
        initial_population = self.prompt_engine.create_initial_population(base_prompt)
        
        # Create agent pool ONCE for all generations
        agent_pool = AgentPool(prompt_debug=self.prompt_debug)
        
        for gen in range(num_generations):
            logger.info(f"\n{'='*50}")
            logger.info(f"GENERATION {gen}")
            logger.info(f"{'='*50}")
            
            # Step 1: Setup agents in existing pool with current population
            
            # Clear previous generation's prompt agents to avoid performance accumulation
            if gen > 0:
                agent_pool.clear_prompt_agents()
                logger.info(f"Cleared prompt agents from previous generation")
            
            # Create prompt agents for all candidates
            model_names = [self.model_name] * len(self.prompt_engine.population)
            self.prompt_engine.setup_agents_in_pool(agent_pool, model_names)
            
            # Step 2: Run tournament
            gen_result = self.tournament.run_generation(generation_id=gen, env_id=self.env_id,agent_pool=agent_pool, rounds=tournament_rounds, phase="evolution", folder_name=f"gen{gen}", replay_buffer=self.replay_buffer, track_tokens=True)

            # log results to wandb
            wandb_log = {
                "generation": gen,
                "tournament/games_played": gen_result.games_played,
                "tournament/population_size": len(self.prompt_engine.population),
                "tokens/total_used": get_total_tokens_used(),
                "tokens/self_play_used": get_total_self_play_tokens_used(),
                "tokens/optimization_used": get_total_optimization_tokens_used(),
            }

            unique_states, unique_trajs = self.track_diversity_from_trajectories(gen_result.trajectories)

            num_games = len(gen_result.trajectories)
            wandb_log.update({
                "diversity_no_buffer/unique_states": unique_states,
                "diversity_no_buffer/unique_trajectories": unique_trajs,
                "diversity_no_buffer/state_diversity_ratio": unique_states / num_games if num_games > 0 else 0,
                "diversity_no_buffer/trajectory_diversity_ratio": unique_trajs / num_games if num_games > 0 else 0,
            })
            
            # Step 3: Determine trajectory path and update memory if enabled
            # Determine trajectory path (used for memory update and evolution)
            trajectory_path = None
            if gen < num_generations - 1:
                if self.trajectories_path:
                    print(f"Using provided trajectories path for evolution: {self.trajectories_path}")
                    trajectory_path = self.trajectories_path
                else:
                    # Only match the evolution trajectory file, not vs_baseline or vs_best files
                    print(f"Using generated trajectories for evolution from gen {gen}")
                    trajectory_path = str((self.output_manager.trajectories_dir / f"gen{gen}_trajectories_gen{gen}_evolution.json").absolute())
            
            # Update memory from generation BEFORE evaluation
            # This ensures the best candidate can access memory during evaluation
            if trajectory_path:
                # Update memory from this generation's games
                self._update_memory_from_generation(gen, agent_pool, trajectory_path)
                
                # Update memory system's current generation
                self.memory_system.current_generation = gen
                self.memory_system._save_current_generation()
            
            # Step 4: Get best candidate and evaluate against baseline
            best_candidate = self.prompt_engine.get_best_candidate(agent_pool, fitness_method=self.fitness_method)
           
            if best_candidate:
                
                self.eval_model_evolution, self.generalize_model_evolution, eval_stats, best_perf, baseline_perf, all_eval_stats, all_eval_performance = self.tournament.run_evaluation(
                    [self.env_id], gen, eval_rounds, best_candidate
                )
                
                # Log eval results to wandb
                self._log_wandb_eval_results(wandb_log, gen, all_eval_stats, all_eval_performance)

                # Add population statistics
                self._log_wandb_population_stats(wandb_log, agent_pool)

                # Save generation result
                result = self._build_generation_result(gen, best_candidate, all_eval_stats, all_eval_performance)
                results.append(result)

            # Step 5: Print summary
            self.prompt_engine.print_summary(agent_pool)
            
            # Step 6: Save generation
            self.prompt_engine.save_generation(agent_pool)
            
            # Print evaluation model evolution after each generation
            if self.eval_model_list and gen > 0:
                if self.generalize_model_list and self.generalize_model_evolution:
                    # Print generalize model evolution with current results
                    # Convert all_eval_performance format for generalize mode
                    converted_perf = {}
                    if all_eval_performance:
                        for model, perf_data in all_eval_performance.items():
                            if isinstance(perf_data, dict) and "vs_eval_models" in perf_data:
                                converted_perf[model] = perf_data
                            else:
                                # Handle format from generalize_model_performance results
                                converted_perf[model] = {
                                    "vs_eval_models": perf_data.get("vs_eval_models", {})
                                }
                    
                    self._print_generalize_model_evolution(final=False, all_eval_stats=all_eval_stats, all_eval_performance=converted_perf)
                elif self.eval_model_evolution:
                    # Print regular eval model evolution
                    self._print_eval_model_evolution(final=False)
            
            # Step 7: Evolve to next generation (if not last)
            if gen < num_generations - 1:
                # Evolve population (trajectory_path already determined in Step 3.5)
                self.prompt_engine.evolve_generation(agent_pool, base_prompt, trajectory_path=trajectory_path)                 
            
            # log into wandb
            wandb.log(wandb_log, step=gen)
        
        # Save final summary
        self.save_evolution_summary(results)
        
        # Print eval model win rate evolution with final summary
        if self.eval_model_list:
            if self.generalize_model_list and self.generalize_model_evolution:
                # Get final generation results for generalize mode summary
                final_eval_stats = None
                final_eval_performance = None
                if results:
                    final_result = results[-1]  # Get last generation's results
                    final_eval_stats = final_result.get("eval_model_list_stats")
                    final_eval_performance = final_result.get("generalize_model_performance")
                    if final_eval_performance:
                        # Convert format to match what _print_generalize_model_evolution expects
                        converted_performance = {}
                        for model, perf_data in final_eval_performance.items():
                            converted_performance[model] = {
                                "vs_eval_models": perf_data.get("vs_eval_models", {})
                            }
                        final_eval_performance = converted_performance
                
                self._print_generalize_model_evolution(final=True, all_eval_stats=final_eval_stats, all_eval_performance=final_eval_performance)
            elif self.eval_model_evolution:
                self._print_eval_model_evolution(final=True)
        
        # Finish wandb
        wandb.finish()
        
        return results
    
    def _log_wandb_eval_results(self, wandb_log: dict, gen: int, all_eval_stats, all_eval_performance):
        """Log evaluation results (generalize or per-env) to wandb_log dict."""
        if not self.eval_model_list or not all_eval_stats:
            return

        if self.generalize_model_list:
            for generalize_model in self.generalize_model_list:
                if generalize_model not in all_eval_stats:
                    continue
                model_stats = all_eval_stats[generalize_model]
                model_performance = all_eval_performance.get(generalize_model, {})
                clean_name = generalize_model.replace("/", "_").replace("-", "_")

                if "vs_baseline" in model_stats:
                    wandb_log[f"generalize/{clean_name}/vs_baseline_games"] = model_stats["vs_baseline"]["games_played"]
                if "vs_evolved" in model_stats:
                    wandb_log[f"generalize/{clean_name}/vs_evolved_games"] = model_stats["vs_evolved"]["games_played"]
                if "generalize_model" in model_performance and model_performance["generalize_model"]:
                    gen_perf = model_performance["generalize_model"]
                    wandb_log[f"generalize/{clean_name}/win_rate"] = gen_perf.win_rate()
                    wandb_log[f"generalize/{clean_name}/trueskill"] = gen_perf.trueskill_rating.mu
                if "vs_eval_models" in model_performance:
                    win_rates = []
                    for eval_model, metrics in model_performance["vs_eval_models"].items():
                        clean_eval = eval_model.replace("/", "_").replace("-", "_")
                        wandb_log[f"generalize/{clean_name}/vs_{clean_eval}/win_rate"] = metrics['win_rate']
                        wandb_log[f"generalize/{clean_name}/vs_{clean_eval}/loss_rate"] = metrics['loss_rate']
                        wandb_log[f"generalize/{clean_name}/vs_{clean_eval}/draw_rate"] = metrics['draw_rate']
                        win_rates.append(metrics['win_rate'])
                    if win_rates:
                        wandb_log[f"generalize/{clean_name}/average_win_rate"] = sum(win_rates) / len(win_rates)
        else:
            clean_env = self.env_id.replace("/", "_").replace("-", "_")
            env_stats = all_eval_stats.get(self.env_id, {})
            env_performance = all_eval_performance.get(self.env_id, {})

            if "vs_baseline" in env_stats:
                wandb_log[f"eval_model_list/{clean_env}/vs_baseline_games"] = env_stats["vs_baseline"]["games_played"]
            if "vs_best" in env_stats:
                wandb_log[f"eval_model_list/{clean_env}/vs_best_games"] = env_stats["vs_best"]["games_played"]
            if "best_candidate" in env_performance and env_performance["best_candidate"]:
                best_cand_perf = env_performance["best_candidate"]
                wandb_log[f"eval_model_list/{clean_env}/best_candidate_win_rate"] = best_cand_perf.win_rate()
                wandb_log[f"eval_model_list/{clean_env}/best_candidate_trueskill"] = best_cand_perf.trueskill_rating.mu

            baseline_win_rates = []
            best_candidate_win_rates = []
            for eval_model in self.eval_model_list:
                evolution_data = self.eval_model_evolution.get(self.env_id, {}).get(eval_model, [])
                clean_model = eval_model.replace("/", "_").replace("-", "_")

                if gen == 0:
                    baseline_entry = next((d for d in evolution_data if d['generation'] == gen and d['opponent'] == 'baseline'), None)
                    if baseline_entry:
                        draw_rate = baseline_entry.get('draw_rate', 0.0)
                        baseline_wr = 1.0 - baseline_entry['win_rate'] - draw_rate
                        wandb_log[f"{clean_env}/baseline_vs_eval_models/{clean_model}"] = baseline_wr
                        baseline_win_rates.append(baseline_wr)

                best_entry = next((d for d in evolution_data if d['generation'] == gen and d['opponent'] == 'best_candidate'), None)
                if best_entry:
                    draw_rate = best_entry.get('draw_rate', 0.0)
                    best_wr = 1.0 - best_entry['win_rate'] - draw_rate
                    wandb_log[f"{clean_env}/best_candidate_vs_eval_models/{clean_model}"] = best_wr
                    best_candidate_win_rates.append(best_wr)

            if baseline_win_rates and gen == 0:
                wandb_log[f"{clean_env}/baseline_vs_eval_models/average"] = sum(baseline_win_rates) / len(baseline_win_rates)
            if best_candidate_win_rates:
                wandb_log[f"{clean_env}/best_candidate_vs_eval_models/average"] = sum(best_candidate_win_rates) / len(best_candidate_win_rates)

    def _log_wandb_population_stats(self, wandb_log: dict, agent_pool):
        """Log population-level statistics to wandb_log dict."""
        rankings = agent_pool.get_ranked_agents()
        if not rankings:
            return

        top_fitness = rankings[0][1]
        avg_fitness = sum(f for _, f in rankings) / len(rankings)
        top_perf = agent_pool.get_agent_performance(rankings[0][0])

        total_wins = sum(agent_pool.get_agent_performance(aid).wins for aid, _ in rankings)
        total_losses = sum(agent_pool.get_agent_performance(aid).losses for aid, _ in rankings)
        total_draws = sum(agent_pool.get_agent_performance(aid).draws for aid, _ in rankings)
        total_games = sum(agent_pool.get_agent_performance(aid).games_played for aid, _ in rankings)

        wandb_log.update({
            "population/best_trueskill": top_fitness,
            "population/avg_trueskill": avg_fitness,
            "population/best_win_rate": top_perf.win_rate() if top_perf else 0,
            "population/best_loss_rate": top_perf.losses / top_perf.games_played if top_perf and top_perf.games_played > 0 else 0,
            "population/best_draw_rate": top_perf.draws / top_perf.games_played if top_perf and top_perf.games_played > 0 else 0,
            "population/avg_win_rate": total_wins / total_games if total_games > 0 else 0,
            "population/avg_loss_rate": total_losses / total_games if total_games > 0 else 0,
            "population/avg_draw_rate": total_draws / total_games if total_games > 0 else 0,
        })

    def _build_generation_result(self, gen: int, best_candidate, all_eval_stats, all_eval_performance) -> dict:
        """Build the result dict for a generation."""
        result = {
            "generation": gen,
            "best_candidate": best_candidate.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        if not self.eval_model_list or not all_eval_stats:
            return result

        result["eval_model_list_stats"] = all_eval_stats
        if not all_eval_performance:
            return result

        if self.generalize_model_list:
            result["generalize_model_performance"] = {}
            for generalize_model, perf_data in all_eval_performance.items():
                model_result = {}
                if "generalize_model" in perf_data and perf_data["generalize_model"]:
                    gen_perf = perf_data["generalize_model"]
                    model_result["overall"] = {
                        "win_rate": gen_perf.win_rate() if gen_perf else 0,
                        "trueskill": gen_perf.trueskill_rating.mu if gen_perf else 25.0,
                        "games_played": gen_perf.games_played if gen_perf else 0,
                    }
                if "vs_eval_models" in perf_data:
                    model_result["vs_eval_models"] = perf_data["vs_eval_models"]
                result["generalize_model_performance"][generalize_model] = model_result
        else:
            if len(all_eval_performance) == 1 and self.env_id in all_eval_performance:
                env_performance = all_eval_performance[self.env_id]
                result["eval_model_list_performance"] = {
                    model: {
                        "win_rate": perf.win_rate() if perf else 0,
                        "trueskill": perf.trueskill_rating.mu if perf else 25.0,
                        "games_played": perf.games_played if perf else 0,
                    } for model, perf in env_performance.items()
                }
            else:
                result["eval_model_list_performance"] = {
                    env_id: {
                        model: {
                            "win_rate": perf.win_rate() if perf else 0,
                            "trueskill": perf.trueskill_rating.mu if perf else 25.0,
                            "games_played": perf.games_played if perf else 0,
                        } for model, perf in env_perf.items()
                    } for env_id, env_perf in all_eval_performance.items()
                }

        return result

    def track_diversity_from_trajectories(self, trajectories: list[list[dict]]):
        """Measures unique game states and unique trajectory prefixes."""
        for trajectory in trajectories:
            for idx, step_data in enumerate(trajectory[:-1]):  # exclude terminal
                # State-level uniqueness
                state_str = json.dumps(step_data["state"], sort_keys=True)
                self.global_seen_states.add(state_str)

                # Trajectory prefix uniqueness (same as ReplayBuffer)
                played_actions = [(t["player_id"], t["action"]) for t in trajectory[: idx + 1]]
                traj_hash = hashlib.sha256(repr(played_actions).encode("utf-8")).hexdigest()

                if traj_hash in self.global_seen_trajectories:
                    continue
                self.global_seen_trajectories.add(traj_hash)

        return len(self.global_seen_states), len(self.global_seen_trajectories)
    
    def _update_memory_from_generation(self, generation: int, agent_pool, trajectory_path: str):
        """Update global memory from this generation's games - handles all memory operations."""
        logger.info(f"Updating global memory from generation {generation} games")
        
        # Create memory subdirectories if they don't exist
        all_insight_dir = self.output_manager.memory_dir / "all_insight"
        all_memory_dir = self.output_manager.memory_dir / "all_memory"
        all_insight_dir.mkdir(exist_ok=True)
        all_memory_dir.mkdir(exist_ok=True)
        
        # Step 1: Load ALL trajectory files for this generation
        logger.info(f"Loading all trajectory files for generation {generation}")
        
        # Find trajectory files for this generation - only evolution files
        trajectory_files = list(self.output_manager.trajectories_dir.glob(f"gen{generation}_trajectories_gen{generation}_evolution.json"))
        if not trajectory_files:
            raise FileNotFoundError(f"No evolution trajectory file found for generation {generation} in {self.output_manager.trajectories_dir}")
        
        # Step 2: Extract reflections from games (limited by max_games_per_agent_reflection if specified)
        all_reflections_data = []
        total_games_processed = 0
        agent_reflection_counts = {}  # Track how many games each agent has reflected on
        
        # Import threading tools at the top of the method
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        # Create a lock for thread-safe operations
        lock = threading.Lock()
        
        def process_single_game(game, game_idx, generation):
            """Process a single game and return reflections if allowed."""
            game_agents = game["agent_names"]
            
            # Filter out baseline agents for limit checking only
            non_baseline_agents = [agent for agent in game_agents if "baseline" not in agent.lower()]
            
            # Check limits with lock to ensure thread safety
            with lock:
                # Check if processing this game would exceed any evolved agent's limit
                should_skip_game = False
                if (self.max_games_per_agent_reflection is not None and 
                    self.max_games_per_agent_reflection > 0):
                    
                    for agent_name in non_baseline_agents:
                        current_count = agent_reflection_counts.get(agent_name, 0)
                        if current_count >= self.max_games_per_agent_reflection:
                            should_skip_game = True
                            break
                
                if should_skip_game:
                    return None  # Skip this game
                
                # Update counts immediately to reserve slots for this game
                for agent_name in non_baseline_agents:
                    agent_reflection_counts[agent_name] = agent_reflection_counts.get(agent_name, 0) + 1
            
            # Extract reflections outside the lock (this is the expensive operation)
            game_reflections = self._extract_game_reflections(game, game_idx, generation)
            return game_reflections
        
        for traj_file in trajectory_files:
            logger.info(f"Processing trajectory file: {traj_file.name}")
            with open(traj_file, 'r') as f:
                trajectories = json.load(f)
            
            # Process games in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                # Submit all games for processing
                futures = []
                for game_idx, game in enumerate(trajectories):
                    future = executor.submit(process_single_game, game, game_idx, generation)
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    game_reflections = future.result()
                    if game_reflections is not None:
                        with lock:
                            all_reflections_data.extend(game_reflections)
                            total_games_processed += 1
        
        if self.prompt_debug:
            logger.info(f"Extracted {len(all_reflections_data)} total reflections from {total_games_processed} games across {len(trajectory_files)} trajectory files")
        
        # Log per-agent reflection counts (evolved agents only - baseline has unlimited)
        if agent_reflection_counts and self.prompt_debug:
            logger.info("Per-agent reflection counts (evolved agents only):")
            for agent_name, count in sorted(agent_reflection_counts.items()):
                logger.info(f"  {agent_name}: {count} games")
            logger.info("Note: Baseline agents generate unlimited reflections (not tracked above)")
        
        # Step 3: Take the ReplayBuffer and create a state abstracted version for memory storage
        all_abstracts_data = []
        replay_buffer_stats = {"total_buffer_size": 0, "requested_topk": 0, "actual_returned": 0}

        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            total_buffer_size = len(self.replay_buffer)
            replay_buffer_stats["total_buffer_size"] = total_buffer_size
            replay_buffer_stats["requested_topk"] = self.replay_topk

            top_states = self.replay_buffer.sample_strategic_states(topk=self.replay_topk)
            replay_buffer_stats["actual_returned"] = len(top_states)

            logger.info(
                f"[ReplayBuffer] Returned {len(top_states)} strategic states "
                f"(out of {total_buffer_size} total buffer entries)"
            )

            # Use the memory system's analyzer LLM for abstract generation
            analyzer_llm = self.memory_system.analyzer.analyzer

            def _generate_single_abstract(state_json_str, stats):
                """Generate an abstract for a single strategic state."""
                wins = stats["wins"]
                losses = stats["losses"]
                draws = stats["draws"]
                last_player_id = stats["player_id"]
                current_player_id = 1 - last_player_id

                abstract_prompt = BASIC_ABSTRACT_GEN_PROMPT.format(
                    strategic_state=state_json_str,
                    wins=wins,
                    losses=losses,
                    draws=draws,
                )
                abstract_text = str(analyzer_llm(abstract_prompt)).strip()
                state_obj = json.loads(state_json_str)

                return {
                    "state": state_obj,
                    "current_player_id": current_player_id,
                    "state_abstract": f"STATE:\n{state_json_str}\n\nABSTRACT: {abstract_text}",
                    "abstract_prompt": abstract_prompt,
                }

            from concurrent.futures import ThreadPoolExecutor as _TPE

            with _TPE(max_workers=self.max_concurrent) as pool:
                futures = [
                    pool.submit(_generate_single_abstract, state_str, stats)
                    for state_str, stats in top_states
                ]
                for fut in futures:
                    result = fut.result()
                    if result is not None:
                        all_abstracts_data.append(result)

            logger.info(f"Generated {len(all_abstracts_data)} state abstracts from replay buffer")

        # Step 4: Save all reflections to all_insight directory
        insight_file = all_insight_dir / f"generation_{generation:02d}_insights.json"
        with open(insight_file, 'w') as f:
            json.dump({
                "generation": generation,
                "timestamp": datetime.now().isoformat(),
                "total_reflections": len(all_reflections_data),
                "reflections": all_reflections_data,
                "state_abstracts": all_abstracts_data,  # Each abstract now contains its own prompt
                "abstract_gen_style": self.abstract_gen_style,  # Log which style was used
                "replay_buffer_stats": replay_buffer_stats  # Log buffer statistics
            }, f, indent=2)
        logger.info(f"Saved raw insights to {insight_file}")
        
        # Step 4: Update global memory using TrajectoryMemorySystem
        self._update_global_memory_from_reflections(all_reflections_data, all_abstracts_data, generation)
        logger.info(f"Updated global memory with {len(all_reflections_data)} game reflections")

    def _extract_game_reflections(self, game: Dict, game_idx: int, generation: int) -> List[Dict]:
        """Extract reflections directly from a game trajectory."""
        reflections_data = []
        
        # Extract game information
        game_id = game.get("game_id", f"gen{generation}_game_{game_idx}")
        agent_names = game["agent_names"]
        
        # Process only non-baseline agents' perspectives
        for agent_idx, agent_name in enumerate(agent_names):
            # Skip baseline agents
            if "baseline" in agent_name.lower():
                continue
            
            # Extract game data for this agent (includes result from outcome)
            game_data = self._extract_agent_game_data(game, agent_idx)
            
            # Use the result that's already computed in game_data
            result = game_data['result']
            
            # Generate reflection using prompt engine
            reflection_data = self.prompt_engine._reflect_on_trajectory_memory(
                game_data,
                "",  # We don't have the original prompt, use empty
                result
            )
            
            # Build reflection entry
            reflection_entry = {
                'candidate_id': agent_name,
                'agent_name': agent_name,
                'game_id': f"{game_id}_agent{agent_idx}",
                'result': result,
                'reflection': reflection_data['reflection'],
                'insight': reflection_data['insight'],
                'reflection_prompt': reflection_data.get('reflection_prompt', ''),
                'format_errors': game_data['format_errors'],
                'total_moves': game_data['total_moves'],
            }
            
            # Add tool fields if tool generation is enabled
            if 'tool' in reflection_data:
                reflection_entry['tool'] = reflection_data['tool']
                reflection_entry['tool_prompt'] = reflection_data.get('tool_prompt', '')
            
            # Add to reflections list
            reflections_data.append(reflection_entry)
        
        return reflections_data
    
    def _extract_agent_game_data(self, game: Dict, agent_idx: int) -> Dict:
        """Extract game data for a specific agent from a game trajectory - simplified version."""
        agent_name = game["agent_names"][agent_idx]
        trajectory = game["trajectory"]
        
        # Extract final state and key metrics
        our_moves = []  # Keep for compatibility with reflection generation
        format_errors = 0
        last_observation = ""
        our_last_action = ""  # Track agent's own last action for reflection
        
        if trajectory:
            # Process trajectory to get final state and agent-specific data
            for step in trajectory:
                is_our_move = step['agent_name'] == agent_name
                
                # Always update last observation to see the complete game progression
                last_observation = step['observation']
                
                if is_our_move:
                    our_moves.append(step['action'])  # Keep for compatibility
                    our_last_action = step['action']  # Track our own last action
                    if not step['format_feedback']['correct_answer_format']:
                        format_errors += 1
        
        # Calculate total moves in the entire game
        total_game_moves = len(trajectory) if trajectory else 0
        
        # Get opponent's agent info
        opponent_name = None
        for name in game["agent_names"]:
            if name != agent_name:
                opponent_name = name
                break
        outcome_reason = game['game_info']["0"]['reason']
        # Simplified game data focusing on final state and summary
        return {
            'last_observation': last_observation,  # Complete final game state
            'last_action': our_last_action,  # The agent's own last action for reflection
            'our_moves': our_moves,  # Agent's moves (kept for compatibility)
            'opponent_name': opponent_name or "Unknown",
            'result': 'win' if game["rewards"][agent_idx] > 0 else ('loss' if game["rewards"][agent_idx] < 0 else 'draw'),
            'format_errors': format_errors,
            'total_moves': total_game_moves,  # Total moves by all players in the game
            'agent_idx': agent_idx,
            'outcome_reason': outcome_reason
        }

    def _update_global_memory_from_reflections(self, reflections_data: List[Dict], abstracts_data: List[Dict], generation: int):
        """Update global memory using TrajectoryMemorySystem."""
        
        # Convert reflections to compressed games format expected by TrajectoryMemorySystem
        compressed_games = []
        
        for reflection in reflections_data:
            # Create a simplified CompressedGame for memory system
            compressed_game = CompressedGame(
                game_id=reflection['game_id'],
                agent_name=reflection['candidate_id'],
                player_id=0,  # Not important for memory
                outcome=1 if reflection['result'] == 'win' else (-1 if reflection['result'] == 'loss' else 0),
                moves=[],  # We don't need moves for simple format
                total_format_errors=reflection['format_errors'],
                total_invalid_moves=0,  # Not tracked in reflections
                game_length=reflection['total_moves']
            )
            compressed_games.append(compressed_game)
        
        # Get current memory from all_memory directory
        all_memory_dir = self.output_manager.memory_dir / "all_memory"
        current_memory = None
        
        # For no_merge, always start with empty memory regardless of generation
        if self.memory_merge_style == "no_merge":
            current_memory = {
                "generation": generation,
                "total_games": 0,
                "performance": {"overall_win_rate": 0.0, "total_wins": 0, "total_losses": 0, "total_draws": 0},
                "format": "simple",
                "insights": [],
                "merge_prompt": "",
                "merge_response": "",
                "state_abstracts": []
            }
        else:
            # Try to load the most recent memory file for other merge styles
            if generation > 0:
                prev_memory_file = all_memory_dir / f"generation_{generation-1:02d}_memory.json"
                if prev_memory_file.exists():
                    with open(prev_memory_file, 'r') as f:
                        current_memory = json.load(f)
            
            if not current_memory:
                # Initialize empty memory
                current_memory = {
                    "generation": generation,
                    "total_games": 0,
                    "performance": {"overall_win_rate": 0.0, "total_wins": 0, "total_losses": 0, "total_draws": 0},
                    "format": "simple",
                    "insights": [],
                    "merge_prompt": "",
                    "merge_response": "",
                    "state_abstracts": []
                }
        
        # Extract insights from reflections (adapt format)
        new_insights = [r['insight'] for r in reflections_data if r['insight']] # type: list[str]
        
        # Extract the state abstracts (now as dicts)
        abstracts = abstracts_data  # type: list[dict]
        
        # Create new analysis dict matching TrajectoryMemorySystem format
        new_analysis = {
            "insights": new_insights
        }

        # Create new abstracts dict
        new_abstracts = {
            "state_abstracts": abstracts
        }
        
        # Calculate performance
        wins = len([r for r in reflections_data if r['result'] == 'win'])
        losses = len([r for r in reflections_data if r['result'] == 'loss'])
        draws = len([r for r in reflections_data if r['result'] == 'draw'])
        total_games = len(reflections_data)
        
        new_performance = {
            "overall_win_rate": wins / total_games if total_games > 0 else 0,
            "total_wins": wins,
            "total_losses": losses,
            "total_draws": draws,
            "avg_format_errors": sum(r['format_errors'] for r in reflections_data) / total_games if total_games > 0 else 0,
            "avg_invalid_moves": 0  # Not tracked
        }
        
        # Use TrajectoryMemorySystem's combine method or direct replacement for no_merge
        if self.memory_merge_style == "no_merge":
            # For no_merge, directly replace with current generation data without combining
            updated_memory = {
                "generation": generation,
                "total_games": total_games,
                "performance": new_performance,
                "format": "simple",
                "insights": new_insights,  # Only current generation insights
                "merge_prompt": f"No merge - using only generation {generation} data",
                "merge_response": f"Direct replacement with {len(new_insights)} insights and {len(abstracts)} state abstracts from generation {generation}",
                "state_abstracts": abstracts  # Only current generation state abstracts
            }
        else:
            # Use TrajectoryMemorySystem's combine method for other merge styles
            updated_memory = self.memory_system._combine_simple_memory(
                current_memory, new_analysis, new_performance, total_games, new_abstracts
            )
        
        # Update generation and save ONCE
        updated_memory["generation"] = generation
        updated_memory["timestamp"] = datetime.now().isoformat()
        
        # Save memory file to all_memory directory
        all_memory_dir = self.output_manager.memory_dir / "all_memory"
        all_memory_dir.mkdir(exist_ok=True)
        memory_file = all_memory_dir / f"generation_{generation:02d}_memory.json"
        with open(memory_file, 'w') as f:
            json.dump(updated_memory, f, indent=2)
        
        logger.info(f"Saved global memory for generation {generation} to {memory_file}")
        
        # Update memory system's generation to keep it in sync
        self.memory_system.current_generation = generation
        self.memory_system._save_current_generation()

    def save_evolution_summary(self, results):
        """Save complete evolution summary."""
        summary = {
            "configuration": {
                "model_name": self.model_name,
                "baseline_model": self.baseline_model,
                "env_id": self.env_id,
                "population_size": self.population_size,
                "keep_ratio": self.keep_ratio,
                "analyzer_model": self.analyzer_model,
                "trajectories_path": self.trajectories_path
            },
            "evolution_results": results,
            "total_generations": len(results),
            "completed_timestamp": datetime.now().isoformat()
        }
        
        # Add eval model evolution data if available
        if self.eval_model_list and self.eval_model_evolution:
            summary["eval_model_evolution"] = self.eval_model_evolution
        
        # Since we removed standard evaluation, we no longer have best_vs_baseline data
        # Improvement tracking based on standard evaluation is no longer available
        
        # Save summary
        summary_file = self.output_manager.summaries_dir / "evolution_summary_final.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nEvolution Summary:")
        logger.info(f"  Completed {len(results)} generations")
        logger.info(f"  Project folder: {self.output_manager.project_root}")
        if self.eval_model_list:
            logger.info(f"  See eval model evolution details above")
    
    def _print_eval_model_evolution(self, final=False):
        """Print win rate evolution for all eval models across generations."""
        print_eval_model_evolution(
            env_id=self.env_id,
            eval_model_list=self.eval_model_list,
            eval_model_evolution=self.eval_model_evolution,
            final=final,
        )

    def _print_generalize_model_evolution(self, final=False, all_eval_stats=None, all_eval_performance=None):
        """Print win rate evolution for generalize models across generations."""
        print_generalize_model_evolution(
            generalize_model_list=self.generalize_model_list,
            eval_model_list=self.eval_model_list,
            generalize_model_evolution=self.generalize_model_evolution,
            final=final,
            all_eval_stats=all_eval_stats,
            all_eval_performance=all_eval_performance,
        )


def main():
    parser = argparse.ArgumentParser(description="Self-Play Prompt Evolution System")
    parser.add_argument("--model", required=True, help="Model to evolve prompts for")
    parser.add_argument("--baseline-model", required=True, help="Baseline model for evaluation")
    parser.add_argument("--base-prompt", required=True, help="Base prompt to start evolution from")
    parser.add_argument("--env", default="TicTacToe-v0", help="Environment")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--tournament-rounds", type=int, default=20, help="Tournament rounds per generation")
    parser.add_argument("--eval-rounds", type=int, default=20, help="Evaluation rounds against baseline")
    parser.add_argument("--population-size", type=int, default=10, help="Population size")
    parser.add_argument("--analyzer-model", default="google/gemini-2.0-flash-001", help="Model for prompt analysis")
    parser.add_argument("--script-name", help="Name for project folder")
    parser.add_argument("--trajectories-path", help="Path to existing trajectory files for learning (glob pattern)")
    parser.add_argument("--max-concurrent", type=int, default=50, help="Maximum concurrent games (default: 50)")
    # Evolution strategy ratios (should sum to 1.0)
    parser.add_argument("--keep-ratio", type=float, default=0.3, help="Ratio of population to keep as elites (default: 0.3)")
    parser.add_argument("--random-ratio", type=float, default=0.2, help="Ratio for pure random exploration without memory (default: 0.2)")
    parser.add_argument("--memory-guided-ratio", type=float, default=0.0, help="Ratio for memory-guided generation using insights (default: 0.0)")
    parser.add_argument("--trajectory-ratio", type=float, default=0.3, help="Ratio for trajectory improvements (default: 0.3)")
    parser.add_argument("--crossover-ratio", type=float, default=0.2, help="Ratio for crossover (default: 0.2)")
    # Memory system settings
    parser.add_argument("--memory-merge-style", default="basic", choices=["simple_add", "no_merge", "basic"], help="Memory merge strategy (default: basic)")
    parser.add_argument("--max-games-in-prompt", type=int, default=1, help="Max games to show in prompt (default: 1)")
    parser.add_argument("--game-sequence-style", default="raw_action_only", help="How to format game sequences (default: raw_action_only)")
    # Reflection settings
    parser.add_argument("--max-games-per-agent-reflection", type=int, default=None, help="Maximum number of games per agent for reflection generation (default: unlimited)")
    # Fitness method for selecting best candidate
    parser.add_argument("--fitness-method", default="trueskill", choices=["trueskill", "winrate"], help="Fitness method for selecting best candidate (default: trueskill)")
    # Evaluation model list
    parser.add_argument("--eval-model-list", nargs="+", help="List of models to evaluate against (e.g., model1 model2 model3)")
    # Generalize model list
    parser.add_argument("--generalize-model-list", nargs="+", help="List of models to test prompt generalization (e.g., model1 model2 model3)")
    # Temperature setting for OpenRouter agents
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model sampling (default: 0.0 for deterministic)")
    # use replay
    parser.add_argument(
        "--use-replay",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="Use a replay buffer (true/false, default: true)"
    )
    parser.add_argument("--beta", type=float, default=0.2, help="Odds of replaying a less recurring trajectory for the tournament (default: 0.2)")
    parser.add_argument("--replay-topk", type=int, default=1, help="Number of top states to sample from replay buffer (default: 1)")
    parser.add_argument("--replay-merge-style", default="basic", choices=["simple_add", "basic"], help="How to merge state abstracts from replay buffer (default: basic)")
    parser.add_argument("--abstract-gen-style", default="basic", choices=["basic"], help="Style for generating state abstracts (default: basic)")
    parser.add_argument("--skip-baseline-eval", action="store_true", help="Skip baseline evaluation (Steps 1 & 2) and set baseline performance to 0 (default: disabled)")
    parser.add_argument("--replay-max-steps", type=int, default=None, help="Maximum number of steps to store per game in replay buffer (default: None = all steps)")
    parser.add_argument("--prompt-debug", action="store_true", help="Enable debug logging for prompts and operations (default: disabled)")
    parser.add_argument("--insight-sampling-mode", default="sample", choices=["partition", "sample", "single"], help="Insight sampling mode for memory-guided generation: partition (non-overlapping), sample (random with overlap), single (one per call) (default: sample)")
    args = parser.parse_args()

    # set seed that will be used in the ta.env.reset()
    # random.seed(77)
    
    evolution = SelfPlayPromptEvolution(
        model_name=args.model,
        baseline_model=args.baseline_model,
        env_id=args.env,
        population_size=args.population_size,
        analyzer_model=args.analyzer_model,
        script_name=args.script_name,
        trajectories_path=args.trajectories_path,
        max_concurrent=args.max_concurrent,
        keep_ratio=args.keep_ratio,
        random_ratio=args.random_ratio,
        memory_guided_ratio=args.memory_guided_ratio,
        trajectory_ratio=args.trajectory_ratio,
        crossover_ratio=args.crossover_ratio,
        memory_merge_style=args.memory_merge_style,
        max_games_in_prompt=args.max_games_in_prompt,
        game_sequence_style=args.game_sequence_style,
        max_games_per_agent_reflection=args.max_games_per_agent_reflection,
        fitness_method=args.fitness_method,
        eval_model_list=args.eval_model_list,
        generalize_model_list=args.generalize_model_list,
        use_replay=args.use_replay,
        beta=args.beta,
        temperature=args.temperature,
        replay_topk=args.replay_topk,
        replay_merge_style=args.replay_merge_style,
        abstract_gen_style=args.abstract_gen_style,
        skip_baseline_eval=args.skip_baseline_eval,
        replay_max_steps=args.replay_max_steps,
        prompt_debug=args.prompt_debug,
        insight_sampling_mode=args.insight_sampling_mode
    )
    
    evolution.run_evolution(
        base_prompt=args.base_prompt,
        num_generations=args.generations,
        tournament_rounds=args.tournament_rounds,
        eval_rounds=args.eval_rounds
    )


if __name__ == "__main__":
    main()