"""
Basic abstract generation prompt for strategic state analysis.
"""

BASIC_ABSTRACT_GEN_PROMPT = """You are analyzing strategically decisive states from this generation's games.
The strategic state gave the biggest variance in wins and loss outcomes within this generation.

STRATEGIC STATE VIEW: {strategic_state}
STRATEGIC STATE OUTCOMES: {wins} wins, {losses} losses, {draws} draws

TASK:
1. Explain why this state is strategically decisive or unusual compared to typical play within 2-3 sentences.
2. Suggest one adjustment or strategic move the player could use when encountering this state in future within 2-3 sentences.

Respond ONLY with plain text (no JSON).
"""
