#!/bin/bash

# Quick Test Script for MPR System
# This is a minimal configuration for testing the system quickly

# Extract script name for logging
SCRIPT_NAME=$(basename "$0" .sh)

# Configuration - Minimal settings for quick testing
MODEL="qwen/qwen-2.5-7b-instruct"
BASELINE_MODEL="qwen/qwen-2.5-7b-instruct"
EVAL_MODEL_LIST="google/gemini-2.5-flash-lite"
BASE_PROMPT="You are playing a two-player zero-sum game. Make valid moves to win. Submit the move enclosed by \\boxed{{}}."
ENV="KuhnPoker-v0-short"

# Quick test parameters
GENERATIONS=2
TOURNAMENT_ROUNDS=5
EVAL_ROUNDS=5
POPULATION_SIZE=5

# Evolution strategy ratios (should sum to 1.0)
KEEP_RATIO=0.25           # % of population kept as elites
RANDOM_RATIO=0.1          # % truly random exploration (no memory)
MEMORY_GUIDED_RATIO=0.5   # % memory-guided prompts (using shared insights)
TRAJECTORY_RATIO=0.0      # % trajectory-based improvements (disabled for quick test)
CROSSOVER_RATIO=0.15      # % crossover

# Memory and replay configuration
MEMORY_MERGE_STYLE="basic"
USE_REPLAY=true
BETA=0.0
REPLAY_TOPK=5
REPLAY_MERGE_STYLE="basic"
ABSTRACT_GEN_STYLE="basic"

# Execution settings
ANALYZER_MODEL="qwen/qwen-2.5-7b-instruct"
MAX_CONCURRENT=20
FITNESS_METHOD="winrate"
TEMPERATURE=0.0
SKIP_BASELINE_EVAL=true
MAX_GAMES_PER_AGENT_REFLECTION="5"
INSIGHT_SAMPLING_MODE="sample"

# Build command flags
SKIP_BASELINE_EVAL_FLAG=""
if [ "$SKIP_BASELINE_EVAL" = "true" ]; then
    SKIP_BASELINE_EVAL_FLAG="--skip-baseline-eval"
fi

# Run prompt evolution
echo "========================================"
echo "MPR Quick Test Script"
echo "========================================"
echo "Model: $MODEL"
echo "Environment: $ENV"
echo "Generations: $GENERATIONS (quick test)"
echo "Population Size: $POPULATION_SIZE (small)"
echo "Tournament Rounds: $TOURNAMENT_ROUNDS (quick)"
echo "Evaluation Rounds: $EVAL_ROUNDS (quick)"
echo ""
echo "Evolution Ratios:"
echo "  Keep:          $KEEP_RATIO"
echo "  Random:        $RANDOM_RATIO"
echo "  Memory-Guided: $MEMORY_GUIDED_RATIO"
echo "  Trajectory:    $TRAJECTORY_RATIO"
echo "  Crossover:     $CROSSOVER_RATIO"
echo ""
echo "Memory: $MEMORY_MERGE_STYLE"
echo "Replay: $USE_REPLAY (β=$BETA, topk=$REPLAY_TOPK)"
echo "========================================"
echo ""

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
echo "Quick test complete!"
echo "Check logs/ directory for results"
