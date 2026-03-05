#!/bin/bash

# =============================================================================
# MPR Quick Start Example Script
# =============================================================================
# This is a simplified configuration for running Multi-Prompt Reasoning (MPR)
# evolution experiments. Adjust parameters below for your use case.
#
# Usage:
#   ./mpr/scripts/run_example.sh
#
# View results:
#   - Check logs/ directory for detailed logs
#   - View wandb dashboard: https://wandb.ai/your-username/prompt-evolution
# =============================================================================

# Extract script name for logging
SCRIPT_NAME=$(basename "$0" .sh)

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# Model Selection
MODEL="qwen/qwen-2.5-7b-instruct"           # Main evolution model
BASELINE_MODEL="qwen/qwen-2.5-7b-instruct"  # Baseline comparison model
EVAL_MODEL_LIST="google/gemini-2.5-flash-lite qwen/qwen3-235b-a22b-2507"  # Models for evaluation

# Game Environment
ENV="KuhnPoker-v0-short"  # Options: KuhnPoker-v0-short, TicTacToe-v0, SimpleNegotiation-v0, etc.

# Base Prompt
BASE_PROMPT="You are playing a two-player zero-sum game. Make valid moves to win. Submit the move enclosed by \\boxed{{}}."

# =============================================================================
# EVOLUTION PARAMETERS
# =============================================================================

GENERATIONS=2              # Number of evolution generations (default: 2 for quick test)
TOURNAMENT_ROUNDS=5        # Games per agent in tournament phase
EVAL_ROUNDS=5              # Games per agent in evaluation phase
POPULATION_SIZE=8          # Number of prompts in population

# Evolution Strategy (must sum to 1.0)
KEEP_RATIO=0.25           # % of top performers kept as elites
RANDOM_RATIO=0.1          # % of truly random exploration (no memory)
MEMORY_GUIDED_RATIO=0.65  # % of memory-guided prompts (using shared insights)
TRAJECTORY_RATIO=0.0      # % based on trajectory analysis
CROSSOVER_RATIO=0.0       # % from crossover of top prompts

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Memory and Reflection
MEMORY_MERGE_STYLE="basic"                   # Memory merge strategy
MAX_GAMES_PER_AGENT_REFLECTION="10"        # Max games for reflection (empty = unlimited)
INSIGHT_SAMPLING_MODE="sample"             # How to sample insights: "sample", "partition", "single"

# Replay Buffer (for advanced use)
USE_REPLAY=false                           # Enable/disable replay mechanism
BETA=0.0                                   # Replay sampling probability (0.0 = disabled)
REPLAY_TOPK=10                             # Top states to sample from buffer
REPLAY_MERGE_STYLE="basic"                 # Replay merge strategy
ABSTRACT_GEN_STYLE="basic"                 # Abstract generation style

# Model Behavior
TEMPERATURE=0.0                            # Sampling temperature (0.0 = deterministic)
FITNESS_METHOD="winrate"                   # Fitness metric: "winrate" or "trueskill"
ANALYZER_MODEL="qwen/qwen-2.5-7b-instruct" # Model for analyzing game trajectories

# Performance
MAX_CONCURRENT=100                         # Maximum concurrent games

# Evaluation
SKIP_BASELINE_EVAL=true                    # Skip baseline evaluation (faster for testing)

# =============================================================================
# EXECUTION
# =============================================================================

# Build command flags
SKIP_BASELINE_EVAL_FLAG=""
if [ "$SKIP_BASELINE_EVAL" = "true" ]; then
    SKIP_BASELINE_EVAL_FLAG="--skip-baseline-eval"
fi

# Print configuration
echo "======================================================================="
echo "MPR: Multi-Prompt Reasoning Evolution"
echo "======================================================================="
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Environment: $ENV"
echo "  Generations: $GENERATIONS"
echo "  Population Size: $POPULATION_SIZE"
echo "  Tournament Rounds: $TOURNAMENT_ROUNDS"
echo "  Evaluation Rounds: $EVAL_ROUNDS"
echo "  Max Concurrent Games: $MAX_CONCURRENT"
echo ""
echo "Evolution Strategy:"
echo "  Keep: ${KEEP_RATIO}, Random: ${RANDOM_RATIO}, Memory-Guided: ${MEMORY_GUIDED_RATIO}, Trajectory: ${TRAJECTORY_RATIO}, Crossover: ${CROSSOVER_RATIO}"
echo ""
echo "Advanced:"
echo "  Memory Merge: $MEMORY_MERGE_STYLE"
echo "  Fitness Method: $FITNESS_METHOD"
echo "  Temperature: $TEMPERATURE"
echo "  Replay Enabled: $USE_REPLAY"
echo "======================================================================="
echo ""

# Run MPR evolution
python -m mpr.self_play_prompt_evolution_memory \
    --model "$MODEL" \
    --baseline-model "$BASELINE_MODEL" \
    --base-prompt "$BASE_PROMPT" \
    --eval-model-list $EVAL_MODEL_LIST \
    --env "$ENV" \
    --generations "$GENERATIONS" \
    --tournament-rounds "$TOURNAMENT_ROUNDS" \
    --eval-rounds "$EVAL_ROUNDS" \
    --population-size "$POPULATION_SIZE" \
    --analyzer-model "$ANALYZER_MODEL" \
    --max-concurrent "$MAX_CONCURRENT" \
    --keep-ratio "$KEEP_RATIO" \
    --random-ratio "$RANDOM_RATIO" \
    --memory-guided-ratio "$MEMORY_GUIDED_RATIO" \
    --trajectory-ratio "$TRAJECTORY_RATIO" \
    --crossover-ratio "$CROSSOVER_RATIO" \
    --memory-merge-style "$MEMORY_MERGE_STYLE" \
    --fitness-method "$FITNESS_METHOD" \
    --temperature "$TEMPERATURE" \
    $SKIP_BASELINE_EVAL_FLAG \
    ${MAX_GAMES_PER_AGENT_REFLECTION:+--max-games-per-agent-reflection "$MAX_GAMES_PER_AGENT_REFLECTION"} \
    --script-name "$SCRIPT_NAME" \
    --use-replay "$USE_REPLAY" \
    --beta "$BETA" \
    --replay-topk "$REPLAY_TOPK" \
    --replay-merge-style "$REPLAY_MERGE_STYLE" \
    --abstract-gen-style "$ABSTRACT_GEN_STYLE" \
    --insight-sampling-mode "$INSIGHT_SAMPLING_MODE"

echo ""
echo "======================================================================="
echo "Evolution Complete!"
echo "Check logs/ directory for detailed results"
echo "View wandb dashboard for visualizations"
echo "======================================================================="
