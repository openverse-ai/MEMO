"""Reporting functions for prompt evolution results.

Extracted from SelfPlayPromptEvolution to keep the main orchestrator focused.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def print_eval_model_evolution(
    env_id: str,
    eval_model_list: List[str],
    eval_model_evolution: Dict,
    final: bool = False,
):
    """Print win rate evolution for all eval models across generations.

    Shows the BEST CANDIDATE's win rate against each eval model over time.

    Args:
        env_id: Environment identifier
        eval_model_list: List of evaluation model names
        eval_model_evolution: Nested dict {env_id: {model: [data_points]}}
        final: If True, print final summary with win rate changes
    """
    logger.info("\n" + "=" * 80)
    logger.info("Best Candidate win rate evolution vs. each evaluation model")
    logger.info("=" * 80)

    eval_envs = [env_id]
    env_baseline_winrates = {}  # {env_id: {model: winrate}}

    for eval_env_id in eval_envs:
        logger.info(f"\n--- Environment: {eval_env_id} ---")

        baseline_winrates = {}

        for eval_model in eval_model_list:
            evolution_data = eval_model_evolution.get(eval_env_id, {}).get(eval_model, [])
            if not evolution_data:
                continue

            win_rates = []
            baseline_winrate = None

            for data in evolution_data:
                if data["opponent"] == "baseline" and baseline_winrate is None:
                    draw_rate = data.get("draw_rate", 0.0)
                    baseline_winrate = 1.0 - data["win_rate"] - draw_rate
                    baseline_winrates[eval_model] = baseline_winrate
                    win_rates.append(f"{baseline_winrate:.1%}")
                elif data["opponent"] == "best_candidate":
                    draw_rate = data.get("draw_rate", 0.0)
                    best_candidate_winrate = 1.0 - data["win_rate"] - draw_rate
                    win_rates.append(f"{best_candidate_winrate:.1%}")

            evolution_str = " -> ".join(win_rates)
            logger.info(f"  Best Candidate vs {eval_model}: {evolution_str}")

        if baseline_winrates:
            env_baseline_winrates[eval_env_id] = baseline_winrates

    if final and env_baseline_winrates:
        logger.info("\n" + "-" * 80)
        logger.info("Finding generation with largest total change from baseline...")

        for eval_env_id in eval_envs:
            if eval_env_id not in env_baseline_winrates:
                continue

            logger.info(f"\n=== Environment: {eval_env_id} ===")
            baseline_winrates = env_baseline_winrates[eval_env_id]

            if baseline_winrates:
                baseline_avg = sum(baseline_winrates.values()) / len(baseline_winrates)
                logger.info(f"Baseline average win rate vs eval models: {baseline_avg:.1%}")

            generation_winrates = {}  # generation -> {model: winrate}
            generation_averages = {}  # generation -> average_winrate

            max_generation = 0
            for eval_model in eval_model_list:
                evolution_data = eval_model_evolution.get(eval_env_id, {}).get(eval_model, [])
                for data in evolution_data:
                    if data["opponent"] == "best_candidate":
                        max_generation = max(max_generation, data["generation"])

            for gen in range(max_generation + 1):
                generation_winrates[gen] = {}

                for eval_model in eval_model_list:
                    if eval_model not in baseline_winrates:
                        continue

                    evolution_data = eval_model_evolution.get(eval_env_id, {}).get(eval_model, [])

                    gen_data = None
                    for data in evolution_data:
                        if data["generation"] == gen and data["opponent"] == "best_candidate":
                            gen_data = data
                            break

                    if gen_data:
                        draw_rate = gen_data.get("draw_rate", 0.0)
                        best_candidate_wr = 1.0 - gen_data["win_rate"] - draw_rate
                        generation_winrates[gen][eval_model] = best_candidate_wr

                if generation_winrates[gen]:
                    generation_averages[gen] = (
                        sum(generation_winrates[gen].values()) / len(generation_winrates[gen])
                    )

            if generation_averages:
                best_gen = max(generation_averages.keys(), key=lambda g: generation_averages[g])
                best_avg_winrate = generation_averages[best_gen]

                logger.info(
                    f"Best candidate best average win rate vs eval models: "
                    f"{best_avg_winrate:.1%} (Generation {best_gen})"
                )
                logger.info(f"\nBest candidate win rates vs each model in Generation {best_gen}:")
                logger.info("-" * 60)

                winrates = generation_winrates[best_gen]
                for eval_model in eval_model_list:
                    if eval_model in winrates:
                        winrate = winrates[eval_model]
                        logger.info(f"  Best Candidate vs {eval_model}: {winrate:.1%}")

    logger.info("=" * 80)


def print_generalize_model_evolution(
    generalize_model_list: List[str],
    eval_model_list: List[str],
    generalize_model_evolution: Dict,
    final: bool = False,
    all_eval_stats: Optional[Dict] = None,
    all_eval_performance: Optional[Dict] = None,
):
    """Print win rate evolution for generalize models across generations.

    Shows how well the evolved prompt generalizes to different models over time.

    Args:
        generalize_model_list: List of generalization model names
        eval_model_list: List of evaluation model names
        generalize_model_evolution: Nested dict {gen_model: {eval_model: [data_points]}}
        final: If True, print final summary with baseline vs best performance comparison
        all_eval_stats: Stats from generalize evaluation tournaments
        all_eval_performance: Performance data from generalize evaluation tournaments
    """
    if not generalize_model_list:
        return

    logger.info("\n" + "=" * 80)
    logger.info("Prompt generalization evolution across different models")
    logger.info("=" * 80)

    if not final:
        logger.info("\nEvolution Progress:")
        logger.info("-" * 80)

        for generalize_model in generalize_model_list:
            logger.info(f"\n{generalize_model}:")

            generation_averages = {}  # {generation: [win_rates]}

            for eval_model in eval_model_list:
                evolution_data = generalize_model_evolution[generalize_model][eval_model]

                win_rates = []
                for data in evolution_data:
                    if data["opponent_type"] == "evolved":
                        gen = data["generation"]
                        win_rate = data["win_rate"]
                        if gen not in generation_averages:
                            generation_averages[gen] = []
                        generation_averages[gen].append(win_rate)
                        win_rates.append(f"Gen{gen}: {win_rate:.1%}")

                if win_rates:
                    evolution_str = " → ".join(win_rates)
                    logger.info(f"  vs {eval_model}: {evolution_str}")

            if generation_averages:
                avg_evolution = []
                for gen in sorted(generation_averages.keys()):
                    avg_rate = sum(generation_averages[gen]) / len(generation_averages[gen])
                    avg_evolution.append(f"Gen{gen}: {avg_rate:.1%}")

                evolution_str = " → ".join(avg_evolution)
                logger.info(f"  Average: {evolution_str}")

        logger.info("=" * 80)
        return

    # Final summary with comprehensive analysis
    logger.info("\nFinal Generalization Evolution Analysis")
    logger.info("=" * 80)

    logger.info("\n1. Complete Evolution History:")
    logger.info("-" * 60)

    model_final_averages = {}  # {generalize_model: final_avg_win_rate}
    model_baseline_averages = {}  # {generalize_model: baseline_avg_win_rate}

    for generalize_model in generalize_model_list:
        logger.info(f"\n{generalize_model}:")

        baseline_rates = []
        final_rates = []
        generation_averages = {}  # {generation: [win_rates]}

        for eval_model in eval_model_list:
            evolution_data = generalize_model_evolution[generalize_model][eval_model]

            baseline_rate = None
            evolved_rates = []

            for data in evolution_data:
                if data["opponent_type"] == "baseline" and data["generation"] == 0:
                    baseline_rate = data["win_rate"]
                    baseline_rates.append(baseline_rate)
                elif data["opponent_type"] == "evolved":
                    gen = data["generation"]
                    win_rate = data["win_rate"]
                    evolved_rates.append((gen, win_rate))
                    if gen not in generation_averages:
                        generation_averages[gen] = []
                    generation_averages[gen].append(win_rate)

            if baseline_rate is not None and evolved_rates:
                evolution_parts = [f"Baseline: {baseline_rate:.1%}"]
                evolution_parts.extend([f"Gen{g}: {r:.1%}" for g, r in evolved_rates])
                evolution_str = " → ".join(evolution_parts)

                final_rate = evolved_rates[-1][1] if evolved_rates else baseline_rate
                final_rates.append(final_rate)
                improvement = final_rate - baseline_rate
                improvement_str = f"({improvement:+.1%})" if improvement != 0 else ""

                logger.info(f"  vs {eval_model}: {evolution_str} {improvement_str}")

        if generation_averages:
            avg_evolution = [
                "Baseline: {:.1%}".format(
                    sum(baseline_rates) / len(baseline_rates) if baseline_rates else 0
                )
            ]
            for gen in sorted(generation_averages.keys()):
                avg_rate = sum(generation_averages[gen]) / len(generation_averages[gen])
                avg_evolution.append(f"Gen{gen}: {avg_rate:.1%}")

            evolution_str = " → ".join(avg_evolution)

            if final_rates:
                final_avg = sum(final_rates) / len(final_rates)
                model_final_averages[generalize_model] = final_avg

            if baseline_rates:
                baseline_avg = sum(baseline_rates) / len(baseline_rates)
                model_baseline_averages[generalize_model] = baseline_avg

                if generalize_model in model_final_averages:
                    improvement = model_final_averages[generalize_model] - baseline_avg
                    improvement_str = f"({improvement:+.1%})" if improvement != 0 else ""
                    logger.info(f"  Average: {evolution_str} {improvement_str}")
                else:
                    logger.info(f"  Average: {evolution_str}")

    # Summary statistics
    logger.info("\n\n2. Summary Statistics:")
    logger.info("-" * 60)

    if model_baseline_averages and model_final_averages:
        overall_baseline = sum(model_baseline_averages.values()) / len(model_baseline_averages)
        logger.info(
            f"Average baseline win rate across all generalize models: {overall_baseline:.1%}"
        )

        overall_final = sum(model_final_averages.values()) / len(model_final_averages)
        logger.info(
            f"Average final win rate across all generalize models: {overall_final:.1%}"
        )

        overall_improvement = overall_final - overall_baseline
        logger.info(f"Overall improvement: {overall_improvement:+.1%}")

        best_model = max(model_final_averages.keys(), key=lambda m: model_final_averages[m])
        best_avg = model_final_averages[best_model]
        best_baseline = model_baseline_averages.get(best_model, 0)
        best_improvement = best_avg - best_baseline
        logger.info(f"\nBest performing generalize model: {best_model}")
        logger.info(
            f"  Final win rate: {best_avg:.1%} (improved {best_improvement:+.1%} from baseline)"
        )

        # Model ranking
        logger.info("\n\n3. Final Model Rankings:")
        logger.info("-" * 60)

        sorted_models = sorted(model_final_averages.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, final_avg) in enumerate(sorted_models, 1):
            baseline_avg = model_baseline_averages.get(model, 0)
            improvement = final_avg - baseline_avg
            logger.info(
                f"{rank}. {model}: {final_avg:.1%} "
                f"(baseline: {baseline_avg:.1%}, improvement: {improvement:+.1%})"
            )

    logger.info("=" * 80)
