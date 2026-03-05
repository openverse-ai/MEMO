#!/usr/bin/env python3
"""
Simple Evaluation: Best Candidate vs Evaluation Model List

This script runs tournaments between a best candidate (with evolved prompt) 
and a list of evaluation models (using base prompt) to measure performance.
Supports multiple runs with statistical analysis.

Features:
- Multiple evaluation runs with --repeat-time
- Statistical analysis (mean ± std) across runs  
- Organized output structure with timestamps
- Separate summary/ and trajectory/ folders

Usage:
    python simple_eval_list.py --evolved-model "google/gemini-2.0-flash-001" \
                               --eval-model-list "Qwen/Qwen3-4B" "claude-3-sonnet" \
                               --evolved-prompt "You are an expert player..." \
                               --base-prompt "You are a TicTacToe player." \
                               --eval-rounds 20 \
                               --repeat-time 3 \
                               --env "TicTacToe-v0"
"""

import argparse
import logging
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import os
import dotenv

dotenv.load_dotenv()

# Reduce httpx logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)

from mpr.tournament.agent_pool import AgentPool
from mpr.evaluation.simple_memory_eval import run_tournament

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Best Candidate vs Evaluation Model List")
    
    # Required arguments
    parser.add_argument("--evolved-model", required=True, help="Model for the best candidate")
    parser.add_argument("--eval-model-list", nargs='+', required=True, help="List of evaluation models")
    parser.add_argument("--evolved-prompt", required=True, help="Evolved prompt for the best candidate")
    parser.add_argument("--eval-rounds", type=int, required=True, help="Number of evaluation rounds")
    
    # Optional arguments
    parser.add_argument("--env", default="TicTacToe-v0", help="Environment (default: TicTacToe-v0)")
    parser.add_argument("--base-prompt", required=True, help="Base prompt for evaluation models")
    parser.add_argument("--repeat-time", type=int, default=1, help="Number of times to repeat the evaluation (default: 1)")
    parser.add_argument("--output-dir", default="evaluation/eval_results", help="Output directory (default: evaluation/eval_results)")
    parser.add_argument("--max-concurrent", type=int, default=50, help="Maximum concurrent games (default: 50)")
    parser.add_argument("--save-trajectories", action="store_true", default=True, help="Save game trajectories (default: true)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model sampling (default: 0.0)")
    
    return parser.parse_args()


def create_evaluation_pool(evolved_model: str, evolved_prompt: str, eval_model_list: List[str], base_prompt: str, temperature: float = 0.0) -> AgentPool:
    """Create evaluation pool with best candidate (using evolved prompt) and evaluation models (using base prompt)."""
    eval_pool = AgentPool()
    
    # Add best candidate first with evolved prompt
    best_candidate_id = f"best_candidate_{evolved_model.split('/')[-1]}"
    eval_pool.add_agent(
        agent_id=best_candidate_id,
        model_name=evolved_model,
        template_name="mpr-evolved",
        evolved_prompt=evolved_prompt,
        temperature=temperature
    )
    logger.info(f"Added best candidate: {best_candidate_id} (model: {evolved_model})")
    
    # Add evaluation models with unique IDs (using base prompt)
    eval_model_to_agent_id = {}
    for i, eval_model in enumerate(eval_model_list):
        eval_agent_id = f"eval_model_{i}_{eval_model.split('/')[-1]}"
        eval_pool.add_agent(
            agent_id=eval_agent_id,
            model_name=eval_model,
            template_name="mpr-evolved",  # Use same template as best candidate
            evolved_prompt=base_prompt,   # Use base prompt for eval models
            temperature=temperature
        )
        eval_model_to_agent_id[eval_model] = eval_agent_id
        logger.info(f"Added eval model: {eval_agent_id} (model: {eval_model}) with base prompt")
    
    return eval_pool, best_candidate_id, eval_model_to_agent_id


def run_evaluation_tournament(eval_pool: AgentPool, env_id: str, eval_rounds: int, 
                            max_concurrent: int, temp_output_dir: str, save_trajectories: bool = False, 
                            temperature: float = 0.0) -> Dict:
    """Run evaluation tournament: eval_model_list vs best_candidate."""
    
    # Create temporary output directory
    output_path = Path(temp_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run tournament with vs_best schedule
    stats = run_tournament(
        agent_pool=eval_pool,
        env_id=env_id,
        num_players_per_game=2,
        num_rounds=eval_rounds,
        max_concurrent=max_concurrent,
        output_dir=str(output_path),
        save_trajectories=save_trajectories,
        memory_agents=[],  # No memory agents
        generation=0,
        phase="eval_vs_best",
        schedule_type="vs_best",  # Use vs_best schedule type
        temperature=temperature
    )
    
    return stats


def create_output_structure(base_output_dir: str) -> Tuple[Path, Path, Path]:
    """Create output directory structure with timestamp."""
    # Get script name and timestamp
    script_name = Path(sys.argv[0]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    main_dir = Path(base_output_dir) / f"{script_name}_{timestamp}"
    summary_dir = main_dir / "summary"
    trajectory_dir = main_dir / "trajectory"
    temp_dir = main_dir / "temp"
    
    # Create directories
    main_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(exist_ok=True)
    trajectory_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created output structure: {main_dir}")
    
    return summary_dir, trajectory_dir, temp_dir


def extract_evaluation_metrics(eval_pool: AgentPool, best_candidate_id: str, eval_model_to_agent_id: Dict[str, str], 
                              eval_model_list: List[str]) -> Dict:
    """Extract evaluation metrics from agent pool."""
    results = {}
    
    # Get best candidate performance
    best_perf = eval_pool.get_agent_performance(best_candidate_id)
    
    # Extract results for each eval model
    for eval_model in eval_model_list:
        agent_id = eval_model_to_agent_id.get(eval_model)
        if agent_id:
            perf = eval_pool.get_agent_performance(agent_id)
            if perf:
                # Calculate rates from eval model's perspective
                eval_loss_rate = perf.losses / perf.games_played if perf.games_played > 0 else 0
                eval_draw_rate = perf.draws / perf.games_played if perf.games_played > 0 else 0
                
                # Invert to show best candidate's perspective
                best_win_rate = 1.0 - perf.win_rate() - eval_draw_rate
                best_loss_rate = perf.win_rate()
                best_draw_rate = eval_draw_rate
                
                results[eval_model] = {
                    "best_win_rate": best_win_rate,
                    "best_loss_rate": best_loss_rate,
                    "best_draw_rate": best_draw_rate,
                    "games_played": perf.games_played
                }
    
    # Calculate overall metrics
    if best_perf:
        overall_loss_rate = best_perf.losses / best_perf.games_played if best_perf.games_played > 0 else 0
        overall_draw_rate = best_perf.draws / best_perf.games_played if best_perf.games_played > 0 else 0
        
        # Calculate average best candidate win rate
        total_best_winrate = sum(r["best_win_rate"] for r in results.values())
        avg_best_winrate = total_best_winrate / len(results) if results else 0
        
        results["overall"] = {
            "win_rate": best_perf.win_rate(),
            "loss_rate": overall_loss_rate,
            "draw_rate": overall_draw_rate,
            "trueskill": best_perf.trueskill_rating.mu,
            "avg_win_rate": avg_best_winrate,
            "games_played": best_perf.games_played
        }
    
    return results


def print_evaluation_results(eval_pool: AgentPool, best_candidate_id: str, eval_model_to_agent_id: Dict[str, str], 
                           evolved_model: str, eval_model_list: List[str], run_number: int = None):
    """Print evaluation results."""
    run_prefix = f"Run {run_number}: " if run_number is not None else ""
    logger.info(f"\n{run_prefix}Evaluation Results:")
    logger.info(f"Best candidate: {best_candidate_id} (model: {evolved_model})")
    logger.info("-" * 80)
    
    # Get best candidate performance
    best_perf = eval_pool.get_agent_performance(best_candidate_id)
    
    # Print results for each eval model
    for eval_model in eval_model_list:
        agent_id = eval_model_to_agent_id.get(eval_model)
        if agent_id:
            perf = eval_pool.get_agent_performance(agent_id)
            if perf:
                # Calculate rates from eval model's perspective
                eval_loss_rate = perf.losses / perf.games_played if perf.games_played > 0 else 0
                eval_draw_rate = perf.draws / perf.games_played if perf.games_played > 0 else 0
                
                # Invert to show best candidate's perspective
                best_win_rate = 1.0 - perf.win_rate() - eval_draw_rate
                best_loss_rate = perf.win_rate()  # When eval wins, best loses
                best_draw_rate = eval_draw_rate    # Draws are the same
                
                logger.info(f"Best Candidate vs {eval_model}: W:{best_win_rate:.1%} L:{best_loss_rate:.1%} D:{best_draw_rate:.1%}")
    
    # Print best candidate overall performance summary
    if best_perf:
        loss_rate = best_perf.losses / best_perf.games_played if best_perf.games_played > 0 else 0
        draw_rate = best_perf.draws / best_perf.games_played if best_perf.games_played > 0 else 0
        
        logger.info("-" * 80)
        logger.info(f"Best Candidate ({evolved_model}) Overall: W:{best_perf.win_rate():.1%} L:{loss_rate:.1%} D:{draw_rate:.1%} (TrueSkill: {best_perf.trueskill_rating.mu:.2f})")
        
        # Calculate average best candidate win rate
        total_best_winrate = 0.0
        count = 0
        for eval_model in eval_model_list:
            agent_id = eval_model_to_agent_id.get(eval_model)
            if agent_id:
                perf = eval_pool.get_agent_performance(agent_id)
                if perf:
                    eval_draw_rate = perf.draws / perf.games_played if perf.games_played > 0 else 0
                    total_best_winrate += (1.0 - perf.win_rate() - eval_draw_rate)
                    count += 1
        
        if count > 0:
            avg_best_winrate = total_best_winrate / count
            logger.info(f"Best Candidate Average Win Rate vs Eval Models: {avg_best_winrate:.1%}")
    
    logger.info("-" * 80)


def calculate_statistics_across_runs(all_run_results: List[Dict], eval_model_list: List[str]) -> Dict:
    """Calculate mean and std of evaluation metrics across all runs."""
    stats = {}
    
    # Process each eval model
    for eval_model in eval_model_list:
        win_rates = []
        loss_rates = []
        draw_rates = []
        
        for run_result in all_run_results:
            if eval_model in run_result:
                win_rates.append(run_result[eval_model]["best_win_rate"])
                loss_rates.append(run_result[eval_model]["best_loss_rate"])
                draw_rates.append(run_result[eval_model]["best_draw_rate"])
        
        if win_rates:
            stats[eval_model] = {
                "win_rate_mean": np.mean(win_rates),
                "win_rate_std": np.std(win_rates),
                "loss_rate_mean": np.mean(loss_rates),
                "loss_rate_std": np.std(loss_rates),
                "draw_rate_mean": np.mean(draw_rates),
                "draw_rate_std": np.std(draw_rates),
            }
    
    # Process overall metrics
    overall_metrics = ["win_rate", "loss_rate", "draw_rate", "trueskill", "avg_win_rate"]
    overall_stats = {}
    
    for metric in overall_metrics:
        values = []
        for run_result in all_run_results:
            if "overall" in run_result and metric in run_result["overall"]:
                values.append(run_result["overall"][metric])
        
        if values:
            overall_stats[f"{metric}_mean"] = np.mean(values)
            overall_stats[f"{metric}_std"] = np.std(values)
    
    stats["overall"] = overall_stats
    return stats


def print_final_statistics(stats: Dict, eval_model_list: List[str], repeat_time: int):
    """Print final statistics across all runs."""
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL STATISTICS ACROSS {repeat_time} RUNS")
    logger.info(f"{'='*80}")
    
    # Print stats for each eval model
    for eval_model in eval_model_list:
        if eval_model in stats:
            model_stats = stats[eval_model]
            logger.info(f"Best Candidate vs {eval_model}:")
            logger.info(f"  Win Rate: {model_stats['win_rate_mean']:.1%} ± {model_stats['win_rate_std']:.1%}")
            logger.info(f"  Loss Rate: {model_stats['loss_rate_mean']:.1%} ± {model_stats['loss_rate_std']:.1%}")
            logger.info(f"  Draw Rate: {model_stats['draw_rate_mean']:.1%} ± {model_stats['draw_rate_std']:.1%}")
            logger.info("")
    
    # Print overall stats
    if "overall" in stats:
        overall = stats["overall"]
        logger.info(f"Overall Performance:")
        if "avg_win_rate_mean" in overall:
            logger.info(f"  Average Win Rate: {overall['avg_win_rate_mean']:.1%} ± {overall['avg_win_rate_std']:.1%}")
        if "trueskill_mean" in overall:
            logger.info(f"  TrueSkill Rating: {overall['trueskill_mean']:.2f} ± {overall['trueskill_std']:.2f}")
    
    logger.info(f"{'='*80}")


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("BEST CANDIDATE VS EVALUATION MODEL LIST")
    logger.info("=" * 60)
    logger.info(f"Evolved Model: {args.evolved_model} (using evolved prompt)")
    logger.info(f"Eval Model List: {', '.join(args.eval_model_list)} (using base prompt)")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Eval Rounds: {args.eval_rounds}")
    logger.info(f"Repeat Times: {args.repeat_time}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info("=" * 60)
    
    try:
        # Create output directory structure
        summary_dir, trajectory_dir, temp_dir = create_output_structure(args.output_dir)
        
        # Store results from all runs
        all_run_results = []
        all_run_summaries = []
        all_trajectories = []
        
        # Run evaluation multiple times
        for run_num in range(1, args.repeat_time + 1):
            logger.info(f"\n{'*'*60}")
            logger.info(f"STARTING RUN {run_num}/{args.repeat_time}")
            logger.info(f"{'*'*60}")
            
            # Create fresh evaluation pool for this run
            eval_pool, best_candidate_id, eval_model_to_agent_id = create_evaluation_pool(
                evolved_model=args.evolved_model,
                evolved_prompt=args.evolved_prompt,
                eval_model_list=args.eval_model_list,
                base_prompt=args.base_prompt,
                temperature=args.temperature
            )
            
            # Create temporary output directory for this run
            run_temp_dir = temp_dir / f"run_{run_num}"
            
            # Run evaluation tournament
            stats = run_evaluation_tournament(
                eval_pool=eval_pool,
                env_id=args.env,
                eval_rounds=args.eval_rounds,
                max_concurrent=args.max_concurrent,
                temp_output_dir=str(run_temp_dir),
                save_trajectories=args.save_trajectories,
                temperature=args.temperature
            )
            
            # Print individual run results
            print_evaluation_results(
                eval_pool=eval_pool,
                best_candidate_id=best_candidate_id,
                eval_model_to_agent_id=eval_model_to_agent_id,
                evolved_model=args.evolved_model,
                eval_model_list=args.eval_model_list,
                run_number=run_num
            )
            
            # Extract metrics for this run
            run_metrics = extract_evaluation_metrics(
                eval_pool=eval_pool,
                best_candidate_id=best_candidate_id,
                eval_model_to_agent_id=eval_model_to_agent_id,
                eval_model_list=args.eval_model_list
            )
            all_run_results.append(run_metrics)
            
            # Collect summary data for this run
            run_summary = {
                "run_number": run_num,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "evolved_model": args.evolved_model,
                    "eval_model_list": args.eval_model_list,
                    "env": args.env,
                    "eval_rounds": args.eval_rounds,
                    "temperature": args.temperature
                },
                "results": run_metrics
            }
            all_run_summaries.append(run_summary)
            
            # Collect trajectory data if enabled
            if args.save_trajectories:
                # Find trajectory files from this run
                traj_files = list(run_temp_dir.glob("trajectories_*.json"))
                for traj_file in traj_files:
                    with open(traj_file, 'r') as f:
                        traj_data = json.load(f)
                        all_trajectories.extend([{**t, "run_number": run_num} for t in traj_data])
        
        # Calculate statistics across all runs
        if args.repeat_time > 1:
            final_stats = calculate_statistics_across_runs(all_run_results, args.eval_model_list)
            print_final_statistics(final_stats, args.eval_model_list, args.repeat_time)
        
        # Save consolidated results
        timestamp = datetime.now().isoformat()
        
        # Save summary.json
        summary_data = {
            "config": {
                "evolved_model": args.evolved_model,
                "eval_model_list": args.eval_model_list,
                "env": args.env,
                "eval_rounds": args.eval_rounds,
                "repeat_time": args.repeat_time,
                "temperature": args.temperature,
                "timestamp": timestamp
            },
            "runs": all_run_summaries
        }
        
        if args.repeat_time > 1:
            summary_data["final_statistics"] = final_stats
        
        summary_file = summary_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")
        
        # Save trajectory.json if trajectories were collected
        if args.save_trajectories and all_trajectories:
            trajectory_data = {
                "config": summary_data["config"],
                "total_trajectories": len(all_trajectories),
                "trajectories": all_trajectories
            }
            trajectory_file = trajectory_dir / "trajectory.json"
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            logger.info(f"Trajectories saved to: {trajectory_file}")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info("✅ Evaluation completed successfully!")
        logger.info(f"Results saved to: {summary_dir.parent}/")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
