#!/bin/bash

# Self-Play Prompt Evolution Example with Enhanced V2 Prompts
# Uses improved abstract generation and merge strategies

# Extract script name for logging
SCRIPT_NAME=$(basename "$0" .sh)

# Replay configuration
USE_REPLAY=true  # Enable/disable replay mechanism (true or false)
BETA=0.4         # Beta parameter for replay mechanism (between 0 and 1)
REPLAY_TOPK=10   # Number of top states to sample from replay buffer
REPLAY_MERGE_STYLE="basic"
ABSTRACT_GEN_STYLE="basic"

# Configuration
MODEL="gpt-4o-mini"
EVAL_MODEL_LIST="google/gemini-2.5-flash-lite"
BASELINE_MODEL="gpt-4o-mini"
BASE_PROMPT="You are playing a two-player zero-sum game. Make valid moves to win.submit the move enclosed by \\boxed{{}}."
ENV="SimpleTak-v0"
GENERATIONS=5

TOURNAMENT_ROUNDS=25
EVAL_ROUNDS=25
POPULATION_SIZE=8

ANALYZER_MODEL="gpt-4o-mini"
MAX_CONCURRENT=50  # Maximum concurrent games
# Optional: Path to existing trajectories for learning
TRAJECTORIES_PATH=""

# Evolution strategy ratios (should sum to 1.0)
KEEP_RATIO=0.25            # % of population kept as elites
RANDOM_RATIO=0.1           # % pure random exploration (no memory)
MEMORY_GUIDED_RATIO=0.55   # % memory-guided generation using insights
TRAJECTORY_RATIO=0.0       # % trajectory-based improvements
CROSSOVER_RATIO=0.1        # % crossover
MEMORY_MERGE_STYLE="no_merge"
SKIP_BASELINE_EVAL=true  # Skip baseline evaluation and set baseline performance to 0
MAX_GAMES_PER_AGENT_REFLECTION="20"  # Maximum number of games per agent for reflection generation (empty = unlimited)
FITNESS_METHOD="winrate"  # Fitness method for selecting best candidate (trueskill or winrate)
TEMPERATURE=1.0 # Temperature for model sampling (0.0 for deterministic)
INSIGHT_SAMPLING_MODE="sample"  # Insight sampling mode: "partition", "sample", or "single"

# Build command with optional memory flags
SKIP_BASELINE_EVAL_FLAG=""

if [ "$SKIP_BASELINE_EVAL" = "true" ]; then
    SKIP_BASELINE_EVAL_FLAG="--skip-baseline-eval"
fi

# Run prompt evolution
echo "Starting Prompt Evolution System with Enhanced V2 Prompts"
echo "========================================================="
echo "Model: $MODEL"
echo "Environment: $ENV"
echo "Generations: $GENERATIONS"
echo "Population Size: $POPULATION_SIZE"
echo "Tournament Rounds: $TOURNAMENT_ROUNDS"
echo "Evaluation Rounds: $EVAL_ROUNDS"
echo "Max Concurrent: $MAX_CONCURRENT"
echo "Evolution Ratios: Keep=$KEEP_RATIO, Random=$RANDOM_RATIO, MemoryGuided=$MEMORY_GUIDED_RATIO, Trajectory=$TRAJECTORY_RATIO, Crossover=$CROSSOVER_RATIO"
echo "Memory Merge Style: $MEMORY_MERGE_STYLE"
echo "Skip Baseline Evaluation: $SKIP_BASELINE_EVAL"
echo "Max Games Per Agent Reflection: ${MAX_GAMES_PER_AGENT_REFLECTION:-'Unlimited'}"
echo "Fitness Method: $FITNESS_METHOD"
echo "Temperature: $TEMPERATURE"
echo "Script Name: $SCRIPT_NAME"
echo "Use Replay: $USE_REPLAY"
echo "Beta: $BETA"
echo "Replay TopK: $REPLAY_TOPK"
echo "Replay Merge Style: $REPLAY_MERGE_STYLE (Enhanced V2)"
echo "Abstract Gen Style: $ABSTRACT_GEN_STYLE (Enhanced V2)"
echo "Insight Sampling Mode: $INSIGHT_SAMPLING_MODE"
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
    ${TRAJECTORIES_PATH:+--trajectories-path "$TRAJECTORIES_PATH"} \
    --use-replay "$USE_REPLAY" \
    --beta "$BETA" \
    --replay-topk "$REPLAY_TOPK" \
    --replay-merge-style "$REPLAY_MERGE_STYLE" \
    --abstract-gen-style "$ABSTRACT_GEN_STYLE" \
    --insight-sampling-mode "$INSIGHT_SAMPLING_MODE"
echo ""
echo "Prompt evolution with enhanced V2 prompts complete!"
echo "Check logs/ directory for results"
