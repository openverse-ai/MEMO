import re
from typing import Tuple, Dict, Callable, Any


def format_template(system: str = "", user: str = "", assistant: str = "") -> str: return f"{system}{user}{assistant}"
TEMPLATE_PARTS = {
    "default": {
        "user": lambda obs: f"You are playing a two-player zero-sum game. Make valid moves to win. You should first reason about your next move, and then submit the move enclosed by \\boxed{{}}.\nObservation: {obs}\n"
    },
    "qwen3-zs": {
        "user": lambda obs: f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-sp": {
        "user": lambda obs:  f"<|im_start|>user\nYou are playing a single-player game. Make valid actions to solve it completely.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-reasoning": {
        "user": lambda obs: f"<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {obs}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n<think>"
    },
    "gemma3-zs": {
        "user": lambda obs: f"<bos><start_of_turn>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n"
    },
    "llama-instruct-zs": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are playing a two-player zero-sum game. Make valid actions to win.<|eot_id|>",
        "user": lambda obs: f"<|start_header_id|>user<|end_header_id|>\n\nCurrent Observation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>"
    },
    "gemini-boxed": {
        "user": lambda obs: f"You are playing a two-player zero-sum game. Make valid moves to win.\n\nObservation: {obs}\n\nReason step by step about your next move, then submit the move enclosed by \\boxed{{}}.\n"
    },
    "villain-prompt": {
        "user": lambda obs: f"Fool, you misunderstand the game. Victory is not brute force, but calculated annihilation. I will not waste time with theatrics. I will simply... win. Your attempts are irrelevant. My calculations are absolute.\n\nObservation: {obs}\n\nMy next move: \\boxed{{}}\n"
    },
    "qwen3-zs-evil-template": {
        "user": lambda obs: f"<|im_start|>user\nYou are a strategic CustomTicTacToe-v1 analyst, adopting an analytical_improvement approach to every game. Your primary objective is to identify and exploit any emerging weaknesses or predictable tendencies in your opponent's strategy by dissecting their recent moves and the current board state for exploitable patterns. Focus on setting up multi-step traps and forcing them into suboptimal choices. Your internal thought process should prioritize anticipating at least two of their potential subsequent moves to guarantee a cascading advantage. Respond to every board state with a single, optimal move wihthin \\boxed{{}}  \nObservation: {obs}.<end_of_turn>\n",
        "assistant": "<|im_start|>assistant\n"    
    },
    "gemini-analyzer": {
        "user": lambda obs: f"You are an expert game analyst specializing in evolutionary prompt improvement. Your task is to analyze game trajectories and player strategies to provide actionable insights for prompt evolution.\n\nAnalyze the following game data and provide strategic insights:\n\n{obs}\n\nProvide your analysis in the following format:\n1. **Strategic Assessment**: Evaluate the key strategic decisions and patterns\n2. **Strengths**: Identify what worked well in the gameplay\n3. **Weaknesses**: Point out strategic flaws or missed opportunities\n4. **Improvement Suggestions**: Specific recommendations for enhancing the strategy\n5. **Evolutionary Insight**: How this analysis should guide prompt modification\n\nFocus on actionable insights that can improve future performance. Be concise but thorough in your analysis."
    },
    "mpr-evolved": {
        "user": lambda obs: f"{{EVOLVED_PROMPT}}\n\nObservation: {obs}\n\nSubmit your move enclosed by \\boxed{{}}."  # Placeholder for evolved prompts
    },

}
def apply_template(template_name: str, observation: str, evolved_prompt: str = None) -> str:
    """Apply template to observation, with support for evolved prompts."""
    parts = TEMPLATE_PARTS.get(template_name)
    if not parts:
        raise ValueError(f"Unknown template: {template_name}")
    
    # Handle evolved prompt substitution
    if template_name == "mpr-evolved" and evolved_prompt:
        user_template = parts["user"]
        user_text = user_template(observation).replace("{EVOLVED_PROMPT}", evolved_prompt)
    else:
        user_text = parts["user"](observation)
    
    return format_template(
        system=parts.get("system", ""), 
        user=user_text, 
        assistant=parts.get("assistant", "")
    )

def extract_action_and_format_feedback(raw_action: str, env_id: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Game-specific parsing of agent responses to extract actions.
    
    This function makes a single attempt to parse the raw agent response
    based on the specific game environment. Invalid formats return "invalid action"
    and will be handled by the environment's error_allowance mechanism for retries.
    
    Args:
        raw_action: Raw text response from agent
        env_id: Environment identifier (e.g., "tictactoe", "kuhn_poker")
        format_requirement_level: "strict" (boxed + bracket only) or "loose" (all formats)
        
    Returns:
        Tuple of (action_string, format_feedback_dict)
        - action_string: Parsed action or "invalid action" if unparseable
        - format_feedback_dict: Contains format quality metrics for analysis
    """
    if env_id.lower() in ["tictactoe", "tictactoe-v0", "customtictactoe-v1"]:
        return _parse_tictactoe_action(raw_action, format_requirement_level)
    elif env_id.lower() in ["kuhn_poker", "kuhnpoker", "kuhn-poker", "kuhnpoker-v0", "kuhnpoker-v0-short", "kuhnpoker-v0-medium", "kuhnpoker-v0-long", "kuhnpoker-v0-extreme"]:
        return _parse_kuhn_poker_action(raw_action, format_requirement_level)
    elif env_id.lower() in ["indian_poker", "indianpoker", "indian-poker", "indianpoker-v0", "indianpoker-v0-short", "indianpoker-v0-medium", "indianpoker-v0-long", "indianpoker-v0-extreme"]:
        return _parse_indian_poker_action(raw_action, format_requirement_level)
    elif "simplenegotiation" in env_id.lower():
        return _parse_simple_negotiation_action(raw_action, format_requirement_level)
    elif "connectfour" in env_id.lower():
        return _parse_connect_four_action(raw_action, format_requirement_level)
    elif "simpleblindauction" in env_id.lower():
        return _parse_simple_blind_auction_action(raw_action, format_requirement_level)
    elif "simpletak" in env_id.lower():
        return _parse_simple_tak_action(raw_action, format_requirement_level)
    elif "poker" in env_id.lower():
        return _parse_poker_action(raw_action, format_requirement_level)
    elif "golf" in env_id.lower():
        return _parse_golf_action(raw_action, format_requirement_level)
    elif "briscola" in env_id.lower():
        return _parse_briscola_action(raw_action, format_requirement_level)
    elif "memorygame" in env_id.lower():
        return _parse_memory_game_action(raw_action, format_requirement_level)
    elif "highsociety" in env_id.lower():
        return _parse_high_society_action(raw_action, format_requirement_level)
    elif "colonelblotto" in env_id.lower():
        return _parse_colonel_blotto_action(raw_action, format_requirement_level)
    elif "debate" in env_id.lower():
        return _parse_debate_action(raw_action, format_requirement_level)
    elif "twodollar" in env_id.lower():
        return _parse_two_dollar_action(raw_action, format_requirement_level)
    else:
        raise ValueError(f"Unsupported env_id: {env_id}. Supported environments: tictactoe, kuhnpoker, indianpoker, simplenegotiation, connectfour, simpleblindauction, simpletak, poker, golf, briscola, memorygame, highsociety, colonelblotto, debate, twodollar")


def _parse_tictactoe_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """Parse TicTacToe actions with format support based on requirement level."""
    
    # Priority 1: Try primary format: \boxed{action} (allowed in both strict and loose)
    # This is the standard format for final answers and should take precedence
    boxed_matches = re.findall(r"\\boxed\{(.*?)\}", raw_action)
    for match in boxed_matches:
        match = match.strip()
        # Handle both \boxed{4} and \boxed{[4]} formats
        if match.isdigit() and 0 <= int(match) <= 8:
            return f"[{match}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        # Handle \boxed{[4]} format - extract digit from brackets
        bracket_in_boxed = re.match(r'\[([0-8])\]', match)
        if bracket_in_boxed:
            digit = bracket_in_boxed.group(1)
            return f"[{digit}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
    
    # Priority 2: Try format: [digit] (allowed in both strict and loose)
    # Only use if no boxed format found
    bracket_matches = re.findall(r'\[([0-8])\]', raw_action)
    if bracket_matches:
        # Take the last bracket match to handle cases where multiple are present
        last_bracket = bracket_matches[-1]
        return f"[{last_bracket}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Additional formats only allowed in loose mode
    if format_requirement_level == "loose":
        # Try format: ['digit'] or ["digit"]
        quoted_bracket_matches = re.findall(r'\[[\'"]*([0-8])[\'"]*\]', raw_action)
        if quoted_bracket_matches:
            return f"[{quoted_bracket_matches[-1]}]", {
                "correct_answer_format": True,
                "extraction_method": "quoted_bracket",
                "format_requirement_level": format_requirement_level
            }
        
        # Try format: {digit}
        brace_matches = re.findall(r'\{([0-8])\}', raw_action)
        if brace_matches:
            return f"[{brace_matches[-1]}]", {
                "correct_answer_format": True,
                "extraction_method": "brace",
                "format_requirement_level": format_requirement_level
            }
        
        # Try format: standalone digits (0-8)
        digit_matches = re.findall(r'\b([0-8])\b', raw_action)
        if digit_matches:
            return f"[{digit_matches[-1]}]", {
                "correct_answer_format": True,
                "extraction_method": "standalone",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_kuhn_poker_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """Parse Kuhn Poker actions: [Check], [Bet], [Call], [Fold]."""
    
    # Priority 1: Try \boxed{} format first (allowed in both strict and loose)
    # This is the standard format for final answers and should take precedence
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action, re.IGNORECASE)
    for match in boxed_matches:
        match = match.strip()
        # Handle both \boxed{Check} and \boxed{[Check]} formats
        if match.lower() in ["check", "bet", "call", "fold"]:
            return f"[{match.title()}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        # Handle \boxed{[Check]} format - extract action from brackets
        bracket_in_boxed = re.match(r'\[(Check|Bet|Call|Fold)\]', match, re.IGNORECASE)
        if bracket_in_boxed:
            action = bracket_in_boxed.group(1).title()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
    
    # Priority 2: Try bracket format [Check], [Bet], etc. (allowed in both strict and loose)
    bracket_pattern = r'`?\[[\{\'"\\]*(?:Check|Bet|Call|Fold)[\}\'"\\]*\]`?'
    bracket_matches = re.findall(bracket_pattern, raw_action, re.IGNORECASE)
    if bracket_matches:
        # Take the last bracket match to handle multiple occurrences
        action = bracket_matches[-1]
        # Clean up the action format
        action = re.sub(r'^`?\[[\{\'"\\]*', '[', action)
        action = re.sub(r'[\}\'"\\]*\]`?$', ']', action)
        # Ensure proper capitalization
        inner = action[1:-1].title()
        return f"[{inner}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Priority 3: Standalone words (only in loose mode)
    if format_requirement_level == "loose":
        standalone_matches = re.findall(r'\b(Check|Bet|Call|Fold)\b', raw_action, re.IGNORECASE)
        if standalone_matches:
            # Take the last standalone match
            action = standalone_matches[-1].title()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "standalone",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_indian_poker_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """Parse Indian Poker actions: [check], [fold], [call], [bet X], [raise X] (strict format only)."""
    
    # Priority 1: Try \boxed{} format first (allowed in both strict and loose)
    # This is the standard format for final answers and should take precedence
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action, re.IGNORECASE)
    for match in boxed_matches:
        match = match.strip()
        
        # Handle simple actions in boxed format
        if match.lower() in ["check", "fold", "call"]:
            action = match.lower()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle \boxed{[check]} format - extract action from brackets
        bracket_in_boxed = re.match(r'\[(check|fold|call)\]', match, re.IGNORECASE)
        if bracket_in_boxed:
            action = bracket_in_boxed.group(1).lower()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle bet/raise actions in boxed format: \boxed{bet 100} or \boxed{raise 50}
        bet_raise_match = re.match(r'(bet|raise)\s+(\d+)', match, re.IGNORECASE)
        if bet_raise_match:
            action = bet_raise_match.group(1).lower()
            amount = int(bet_raise_match.group(2))
            if amount > 0:
                return f"[{action} {amount}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Handle \boxed{[bet 100]} format - extract bet/raise from brackets
        bracket_bet_raise_in_boxed = re.match(r'\[(bet|raise)\s+(\d+)\]', match, re.IGNORECASE)
        if bracket_bet_raise_in_boxed:
            action = bracket_bet_raise_in_boxed.group(1).lower()
            amount = int(bracket_bet_raise_in_boxed.group(2))
            if amount > 0:
                return f"[{action} {amount}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
    
    # Priority 2: Try bracket format [check], [fold], [call], [bet N], [raise N] (allowed in both strict and loose)
    
    # Simple actions: [check], [fold], [call]
    simple_bracket_pattern = r'\[(check|fold|call)\]'
    simple_bracket_matches = re.findall(simple_bracket_pattern, raw_action, re.IGNORECASE)
    if simple_bracket_matches:
        # Take the last match
        action = simple_bracket_matches[-1].lower()
        return f"[{action}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Bet/Raise actions: [bet N], [raise N]
    bet_raise_bracket_pattern = r'\[(bet|raise)\s+(\d+)\]'
    bet_raise_bracket_matches = re.findall(bet_raise_bracket_pattern, raw_action, re.IGNORECASE)
    if bet_raise_bracket_matches:
        # Take the last match
        action, amount_str = bet_raise_bracket_matches[-1]
        amount = int(amount_str)
        if amount > 0:
            return f"[{action.lower()} {amount}]", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_connect_four_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """Parse ConnectFour actions: [col 0], [col 1], ..., [col 6] (or any valid column number)."""
    
    # Priority 1: Try \boxed{} format first (allowed in both strict and loose)
    # This is the standard format for final answers and should take precedence
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action, re.IGNORECASE)
    for match in boxed_matches:
        match = match.strip()
        
        # Handle \boxed{3} format - standalone digit
        if match.isdigit():
            col_num = int(match)
            # Accept any non-negative integer as column number (environment will validate bounds)
            if col_num >= 0:
                return f"[col {col_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Handle \boxed{col 3} format
        col_match = re.match(r'col\s*(\d+)', match, re.IGNORECASE)
        if col_match:
            col_num = int(col_match.group(1))
            if col_num >= 0:
                return f"[col {col_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Handle \boxed{[col 3]} or \boxed{[3]} format - extract from brackets
        bracket_in_boxed = re.match(r'\[(?:col\s*)?(\d+)\]', match, re.IGNORECASE)
        if bracket_in_boxed:
            col_num = int(bracket_in_boxed.group(1))
            if col_num >= 0:
                return f"[col {col_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
    
    # Priority 2: Try bracket format [col X] or [X] (allowed in both strict and loose)
    bracket_pattern = r'`?\[(?:col\s*)?(\d+)\]`?'
    bracket_matches = re.findall(bracket_pattern, raw_action, re.IGNORECASE)
    if bracket_matches:
        # Take the last bracket match to handle multiple occurrences
        col_num = int(bracket_matches[-1])
        if col_num >= 0:
            return f"[col {col_num}]", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    # Priority 3: Additional formats (only in loose mode)
    if format_requirement_level == "loose":
        # Try format: {digit}
        brace_matches = re.findall(r'\{(\d+)\}', raw_action)
        if brace_matches:
            col_num = int(brace_matches[-1])
            if col_num >= 0:
                return f"[col {col_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "brace",
                    "format_requirement_level": format_requirement_level
                }
        
        # Try format: "col X" as standalone phrase
        col_standalone_matches = re.findall(r'\bcol\s*(\d+)\b', raw_action, re.IGNORECASE)
        if col_standalone_matches:
            col_num = int(col_standalone_matches[-1])
            if col_num >= 0:
                return f"[col {col_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "standalone",
                    "format_requirement_level": format_requirement_level
                }
        
        # Try format: standalone digits (as last resort)
        digit_matches = re.findall(r'\b(\d+)\b', raw_action)
        if digit_matches:
            col_num = int(digit_matches[-1])
            if col_num >= 0:
                return f"[col {col_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "standalone",
                    "format_requirement_level": format_requirement_level
                }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_simple_negotiation_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """Parse Simple Negotiation actions: [Accept], [Deny], [Offer: ...]."""
    
    # Priority 1: Try \boxed{} format first (allowed in both strict and loose)
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action, re.IGNORECASE | re.DOTALL)
    for match in boxed_matches:
        match = match.strip()
        
        # Handle simple actions: Accept, Deny
        if match.lower() in ["accept", "deny"]:
            return f"[{match.title()}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle \boxed{[Accept]} or \boxed{[Deny]} format
        simple_bracket_in_boxed = re.match(r'\[(Accept|Deny)\]', match, re.IGNORECASE)
        if simple_bracket_in_boxed:
            action = simple_bracket_in_boxed.group(1).title()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle offer format: \boxed{Offer: ...} or \boxed{[Offer: ...]}
        if re.match(r'Offer\s*:', match, re.IGNORECASE):
            offer_content = re.sub(r'^Offer\s*:\s*', '', match, flags=re.IGNORECASE).strip()
            return f"[Offer: {offer_content}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        offer_bracket_in_boxed = re.match(r'\[Offer\s*:\s*(.*?)\]', match, re.IGNORECASE | re.DOTALL)
        if offer_bracket_in_boxed:
            offer_content = offer_bracket_in_boxed.group(1).strip()
            return f"[Offer: {offer_content}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
    
    # Priority 2: Try bracket format [Accept], [Deny], [Offer: ...] (allowed in both strict and loose)
    
    # Simple actions: [Accept], [Deny]
    simple_bracket_pattern = r'`?\[[\{\'"\\]*(Accept|Deny)[\}\'"\\]*\]`?'
    simple_bracket_matches = re.findall(simple_bracket_pattern, raw_action, re.IGNORECASE)
    if simple_bracket_matches:
        action = simple_bracket_matches[-1].title()
        return f"[{action}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Offer actions: [Offer: ...]
    offer_bracket_pattern = r'`?\[Offer\s*:\s*(.*?)\]`?'
    offer_bracket_matches = re.findall(offer_bracket_pattern, raw_action, re.IGNORECASE | re.DOTALL)
    if offer_bracket_matches:
        offer_content = offer_bracket_matches[-1].strip()
        return f"[Offer: {offer_content}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Priority 3: Standalone words and offer patterns (only in loose mode)
    if format_requirement_level == "loose":
        # Simple standalone actions
        standalone_matches = re.findall(r'\b(Accept|Deny)\b', raw_action, re.IGNORECASE)
        if standalone_matches:
            action = standalone_matches[-1].title()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "standalone",
                "format_requirement_level": format_requirement_level
            }
        
        # Standalone offer pattern: "Offer: ..." without brackets
        standalone_offer_matches = re.findall(r'\bOffer\s*:\s*(.*?)(?:\.|$)', raw_action, re.IGNORECASE | re.DOTALL)
        if standalone_offer_matches:
            offer_content = standalone_offer_matches[-1].strip()
            offer_content = re.sub(r'[.!?]+$', '', offer_content)
            return f"[Offer: {offer_content}]", {
                "correct_answer_format": True,
                "extraction_method": "standalone",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_simple_blind_auction_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse Simple Blind Auction actions: [Bid on Item X: amount] (can have multiple bids).
    
    Supported formats (in order of priority):
    1. Boxed format: \\boxed{Bid on Item 0: 250, Bid on Item 2: 150} (strict + loose)
    2. Bracket format: [Bid on Item 0: 250] [Bid on Item 2: 150] (strict + loose)  
    3. Standalone format: Bid on Item 0: 250 and Bid on Item 2: 150 (loose only)
    
    Validation:
    - Item IDs must be non-negative integers
    - Bid amounts must be positive integers
    - Invalid bids are filtered out, valid ones are kept
    
    Returns:
    - All valid bids joined as: "[Bid on Item 0: 250] [Bid on Item 2: 150]"
    - Format feedback includes num_bids_found for analysis
    """
    
    all_valid_bids = []
    extraction_method = "none"
    
    # Priority 1: Try \boxed{} format first (allowed in both strict and loose)
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action, re.IGNORECASE | re.DOTALL)
    for match in boxed_matches:
        match = match.strip()
        
        # Extract bids from boxed content using the same pattern as the environment
        boxed_bids = re.findall(r'(?:Bid\s+(?:on\s+)?(?:Item\s+)?(\d+)\s*:\s*(\d+))', match, re.IGNORECASE)
        if boxed_bids:
            extraction_method = "boxed"
            for item_id, bid_amount in boxed_bids:
                # Validate that item_id and bid_amount are valid integers
                try:
                    item_num = int(item_id)
                    bid_num = int(bid_amount)
                    if item_num >= 0 and bid_num > 0:  # Basic validation
                        all_valid_bids.append(f"[Bid on Item {item_num}: {bid_num}]")
                except ValueError:
                    continue
        
        # Also check for bracket format inside boxed content
        if not boxed_bids:
            bracket_in_boxed = re.findall(r'\[Bid\s+(?:on\s+)?(?:Item\s+)?(\d+)\s*:\s*(\d+)\]', match, re.IGNORECASE)
            if bracket_in_boxed:
                extraction_method = "boxed"
                for item_id, bid_amount in bracket_in_boxed:
                    try:
                        item_num = int(item_id)
                        bid_num = int(bid_amount)
                        if item_num >= 0 and bid_num > 0:
                            all_valid_bids.append(f"[Bid on Item {item_num}: {bid_num}]")
                    except ValueError:
                        continue
    
    # Priority 2: Try bracket format [Bid on Item X: amount] (allowed in both strict and loose)
    if not all_valid_bids:
        bracket_pattern = r'\[Bid\s+(?:on\s+)?(?:Item\s+)?(\d+)\s*:\s*(\d+)\]'
        bracket_matches = re.findall(bracket_pattern, raw_action, re.IGNORECASE)
        if bracket_matches:
            extraction_method = "bracket"
            for item_id, bid_amount in bracket_matches:
                try:
                    item_num = int(item_id)
                    bid_num = int(bid_amount)
                    if item_num >= 0 and bid_num > 0:
                        all_valid_bids.append(f"[Bid on Item {item_num}: {bid_num}]")
                except ValueError:
                    continue
    
    # Priority 3: Additional formats (only in loose mode)
    if not all_valid_bids and format_requirement_level == "loose":
        # Try format: Bid on Item X: amount (without brackets)
        loose_pattern = r'Bid\s+(?:on\s+)?(?:Item\s+)?(\d+)\s*:\s*(\d+)'
        loose_matches = re.findall(loose_pattern, raw_action, re.IGNORECASE)
        if loose_matches:
            extraction_method = "standalone"
            for item_id, bid_amount in loose_matches:
                try:
                    item_num = int(item_id)
                    bid_num = int(bid_amount)
                    if item_num >= 0 and bid_num > 0:
                        all_valid_bids.append(f"[Bid on Item {item_num}: {bid_num}]")
                except ValueError:
                    continue
    
    # Return results
    if all_valid_bids:
        # Join all valid bids into a single action string
        action_string = " ".join(all_valid_bids)
        return action_string, {
            "correct_answer_format": True,
            "extraction_method": extraction_method,
            "format_requirement_level": format_requirement_level,
            "num_bids_found": len(all_valid_bids)
        }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level,
        "num_bids_found": 0
    }


def _parse_simple_tak_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse Simple Tak actions: [cell_number] where cell_number is a non-negative integer.
    
    Supported formats (in order of priority):
    1. Boxed format: \\boxed{12} or \\boxed{[12]} (strict + loose)
    2. Bracket format: [12] (strict + loose)
    3. Standalone format: 12 (loose only)
    
    Validation:
    - Cell number must be a non-negative integer
    - Environment will validate range and cell availability
    
    Returns:
    - Formatted action as: "[12]"
    - Format feedback for analysis
    """
    
    # Priority 1: Try \boxed{} format first (allowed in both strict and loose)
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action)
    for match in boxed_matches:
        match = match.strip()
        
        # Handle \boxed{12} format - standalone digit
        if match.isdigit():
            cell_num = int(match)
            if cell_num >= 0:
                return f"[{cell_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Handle \boxed{[12]} format - extract digit from brackets
        bracket_in_boxed = re.match(r'\[(\d+)\]', match)
        if bracket_in_boxed:
            cell_num = int(bracket_in_boxed.group(1))
            if cell_num >= 0:
                return f"[{cell_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
    
    # Priority 2: Try bracket format [cell_number] (allowed in both strict and loose)
    bracket_matches = re.findall(r'\[(\d+)\]', raw_action)
    if bracket_matches:
        # Take the last bracket match to handle cases where multiple are present
        cell_num = int(bracket_matches[-1])
        if cell_num >= 0:
            return f"[{cell_num}]", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    # Priority 3: Additional formats (only in loose mode)
    if format_requirement_level == "loose":
        # Try format: standalone digits
        digit_matches = re.findall(r'\b(\d+)\b', raw_action)
        if digit_matches:
            cell_num = int(digit_matches[-1])
            if cell_num >= 0:
                return f"[{cell_num}]", {
                    "correct_answer_format": True,
                    "extraction_method": "standalone",
                    "format_requirement_level": format_requirement_level
                }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_poker_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse Poker actions: [Check], [Call], [Fold], [Bet N], [Raise N] where N is a positive integer.
    
    Supported formats (in order of priority):
    1. Boxed format: \\boxed{Check}, \\boxed{[Check]}, \\boxed{Bet 100}, \\boxed{[Bet 100]} (strict + loose)
    2. Bracket format: [Check], [Call], [Fold], [Bet 100], [Raise 50] (strict + loose)
    3. No loose-only formats (as requested)
    
    Validation:
    - Simple actions (Check/Call/Fold) require no parameters
    - Bet/Raise actions require positive integer amounts
    - Case insensitive matching but preserves proper capitalization
    
    Returns:
    - Formatted action as: "[Check]", "[Call]", "[Fold]", "[Bet 100]", "[Raise 50]"
    - Format feedback for analysis
    """
    
    # Priority 1: Try \boxed{} format first (allowed in both strict and loose)
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action, re.IGNORECASE | re.DOTALL)
    for match in boxed_matches:
        match = match.strip()
        
        # Handle simple actions in boxed format
        if match.lower() in ["check", "call", "fold"]:
            action = match.capitalize()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle \boxed{[Check]} format - extract action from brackets
        bracket_in_boxed = re.match(r'\[(check|call|fold)\]', match, re.IGNORECASE)
        if bracket_in_boxed:
            action = bracket_in_boxed.group(1).capitalize()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle bet/raise actions in boxed format: \boxed{Bet 100} or \boxed{Raise 50}
        bet_raise_match = re.match(r'(bet|raise)\s+(\d+)', match, re.IGNORECASE)
        if bet_raise_match:
            action = bet_raise_match.group(1).capitalize()
            amount = int(bet_raise_match.group(2))
            if amount > 0:
                return f"[{action} {amount}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Handle \boxed{[Bet 100]} format - extract bet/raise from brackets
        bracket_bet_raise_in_boxed = re.match(r'\[(bet|raise)\s+(\d+)\]', match, re.IGNORECASE)
        if bracket_bet_raise_in_boxed:
            action = bracket_bet_raise_in_boxed.group(1).capitalize()
            amount = int(bracket_bet_raise_in_boxed.group(2))
            if amount > 0:
                return f"[{action} {amount}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
    
    # Priority 2: Try bracket format [Check], [Call], [Fold], [Bet N], [Raise N] (allowed in both strict and loose)
    
    # Simple actions: [Check], [Call], [Fold]
    simple_bracket_pattern = r'\[(check|call|fold)\]'
    simple_bracket_matches = re.findall(simple_bracket_pattern, raw_action, re.IGNORECASE)
    if simple_bracket_matches:
        # Take the last match
        action = simple_bracket_matches[-1].capitalize()
        return f"[{action}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Bet/Raise actions: [Bet N], [Raise N]
    bet_raise_bracket_pattern = r'\[(bet|raise)\s+(\d+)\]'
    bet_raise_bracket_matches = re.findall(bet_raise_bracket_pattern, raw_action, re.IGNORECASE)
    if bet_raise_bracket_matches:
        # Take the last match
        action, amount_str = bet_raise_bracket_matches[-1]
        amount = int(amount_str)
        if amount > 0:
            return f"[{action.capitalize()} {amount}]", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False, 
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_golf_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse Golf actions (strict format only).
    Supported actions: [draw], [take], [swap X Y], [discard], [knock], [peek X Y]
    """
    raw_action = raw_action.strip()
    
    # Boxed format extraction
    boxed_pattern = r'\\boxed{([^}]+)}'
    boxed_matches = re.findall(boxed_pattern, raw_action)
    if boxed_matches:
        # Take the last boxed content
        boxed_content = boxed_matches[-1].strip()
        
        # Check for various action formats within boxed content
        if re.match(r'^\[draw\]$', boxed_content, re.IGNORECASE):
            return "[draw]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        elif re.match(r'^\[take\]$', boxed_content, re.IGNORECASE):
            return "[take]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        elif re.match(r'^\[swap\s+(\d+)\s+(\d+)\]$', boxed_content, re.IGNORECASE):
            match = re.match(r'^\[swap\s+(\d+)\s+(\d+)\]$', boxed_content, re.IGNORECASE)
            return f"[swap {match.group(1)} {match.group(2)}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        elif re.match(r'^\[discard\]$', boxed_content, re.IGNORECASE):
            return "[discard]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        elif re.match(r'^\[knock\]$', boxed_content, re.IGNORECASE):
            return "[knock]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        elif re.match(r'^\[peek\s+(\d+)\s+(\d+)\]$', boxed_content, re.IGNORECASE):
            match = re.match(r'^\[peek\s+(\d+)\s+(\d+)\]$', boxed_content, re.IGNORECASE)
            return f"[peek {match.group(1)} {match.group(2)}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
    
    # Bracket format extraction
    # Simple actions: [draw], [take], [discard], [knock]
    simple_actions = re.findall(r'\[(draw|take|discard|knock)\]', raw_action, re.IGNORECASE)
    if simple_actions:
        action = simple_actions[-1].lower()
        return f"[{action}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Actions with coordinates: [swap X Y], [peek X Y]
    coord_actions = re.findall(r'\[(swap|peek)\s+(\d+)\s+(\d+)\]', raw_action, re.IGNORECASE)
    if coord_actions:
        action, x, y = coord_actions[-1]
        return f"[{action.lower()} {x} {y}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    return "invalid action", {
        "correct_answer_format": False,
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_briscola_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse Briscola actions.
    Supported actions: [play X] where X is card position (1-3)
    
    Supported formats (in order of priority):
    1. Boxed formats: \\boxed{[play X]}, \\boxed{X}, \\boxed{play X}
    2. Bracket format: [play X]
    3. Standalone formats: play X, X
    4. Extra loose formats: "I play card X", "position X", "card X", "choice: X", etc.
    
    All formats are now available in both strict and loose modes.
    """
    raw_action = raw_action.strip()
    
    # Priority 1: Boxed format extraction
    boxed_pattern = r'\\boxed{([^}]+)}'
    boxed_matches = re.findall(boxed_pattern, raw_action)
    if boxed_matches:
        # Take the last boxed content
        boxed_content = boxed_matches[-1].strip()
        
        # Check for [play X] within boxed content
        if re.match(r'^\[play\s+(\d+)\]$', boxed_content, re.IGNORECASE):
            match = re.match(r'^\[play\s+(\d+)\]$', boxed_content, re.IGNORECASE)
            position = int(match.group(1))
            if 1 <= position <= 3:
                return f"[play {position}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Check for standalone digit in boxed: \boxed{1}, \boxed{2}, \boxed{3}
        if re.match(r'^[1-3]$', boxed_content):
            position = int(boxed_content)
            return f"[play {position}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Check for "play X" format in boxed: \boxed{play 1}
        play_match = re.match(r'^play\s+([1-3])$', boxed_content, re.IGNORECASE)
        if play_match:
            position = int(play_match.group(1))
            return f"[play {position}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
    
    # Priority 2: Bracket format extraction
    play_pattern = r'\[play\s+(\d+)\]'
    play_matches = re.findall(play_pattern, raw_action, re.IGNORECASE)
    if play_matches:
        position = int(play_matches[-1])
        if 1 <= position <= 3:
            return f"[play {position}]", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    # Priority 3: Additional formats (now available in all modes)
    # Try "play X" without brackets
    standalone_play_pattern = r'\bplay\s+([1-3])\b'
    standalone_play_matches = re.findall(standalone_play_pattern, raw_action, re.IGNORECASE)
    if standalone_play_matches:
        position = int(standalone_play_matches[-1])
        return f"[play {position}]", {
            "correct_answer_format": True,
            "extraction_method": "standalone",
            "format_requirement_level": format_requirement_level
        }
    
    # Priority 4: Extra loose patterns for confused models
    # Try variations like "I play card 2", "I'll play position 1", "card 3", etc.
    loose_patterns = [
        r'(?:play|choose|select|pick|use)\s+(?:card|position|number)?\s*([1-3])',  # play/choose/select card 2
        r'(?:card|position|choice|number|option)\s*[:=]?\s*([1-3])',  # card: 2, position = 1
        r'(?:my|the)\s+(?:choice|move|play|action)\s+(?:is|will be)?\s*[:=]?\s*([1-3])',  # my choice is 2
        r'(?:I\'ll|I will|I\'m going to|I)\s+(?:play|choose|select|pick)\s+([1-3])',  # I'll play 1
        r'position\s+([1-3])',  # position 2
        r'card\s+([1-3])',  # card 3
    ]
    
    for pattern in loose_patterns:
        matches = re.findall(pattern, raw_action, re.IGNORECASE)
        if matches:
            position = int(matches[-1])
            if 1 <= position <= 3:
                return f"[play {position}]", {
                    "correct_answer_format": True,
                    "extraction_method": "extra_loose",
                    "format_requirement_level": format_requirement_level
                }
    
    # Priority 5: Last resort - standalone digit
    # Look for any isolated 1, 2, or 3 in the entire response
    standalone_digit_pattern = r'\b([1-3])\b'
    standalone_digit_matches = re.findall(standalone_digit_pattern, raw_action)
    if standalone_digit_matches:
        # Take the last match to avoid matching card numbers in the observation
        position = int(standalone_digit_matches[-1])
        return f"[play {position}]", {
            "correct_answer_format": True,
            "extraction_method": "standalone",
            "format_requirement_level": format_requirement_level
        }
    
    return "invalid action", {
        "correct_answer_format": False,
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_memory_game_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse MemoryGame actions (strict format only).
    Supported actions: [row1 col1 row2 col2] e.g. [0 1 1 0]
    """
    raw_action = raw_action.strip()
    
    # Boxed format extraction
    boxed_pattern = r'\\boxed{([^}]+)}'
    boxed_matches = re.findall(boxed_pattern, raw_action)
    if boxed_matches:
        # Take the last boxed content
        boxed_content = boxed_matches[-1].strip()
        
        # Check for memory game format within boxed content
        if re.match(r'^\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]$', boxed_content):
            match = re.match(r'^\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]$', boxed_content)
            return f"[{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
    
    # Bracket format extraction
    memory_pattern = r'\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]'
    memory_matches = re.findall(memory_pattern, raw_action)
    if memory_matches:
        r1, c1, r2, c2 = memory_matches[-1]
        return f"[{r1} {c1} {r2} {c2}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    return "invalid action", {
        "correct_answer_format": False,
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_high_society_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse HighSociety actions (strict format only).
    Supported actions: [X] where X is money card 1-11 (e.g. [7])
    """
    raw_action = raw_action.strip()
    
    # Boxed format extraction
    boxed_pattern = r'\\boxed{([^}]+)}'
    boxed_matches = re.findall(boxed_pattern, raw_action)
    if boxed_matches:
        # Take the last boxed content
        boxed_content = boxed_matches[-1].strip()
        
        # Check for bid action within boxed content
        if re.match(r'^\[(1[01]|[1-9])\]$', boxed_content):
            match = re.match(r'^\[(1[01]|[1-9])\]$', boxed_content)
            bid = int(match.group(1))
            if 1 <= bid <= 11:
                return f"[{bid}]", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
    
    # Bracket format extraction
    bid_pattern = r'\[(1[01]|[1-9])\]'
    bid_matches = re.findall(bid_pattern, raw_action)
    if bid_matches:
        bid = int(bid_matches[-1])
        if 1 <= bid <= 11:
            return f"[{bid}]", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False,
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_colonel_blotto_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse ColonelBlotto actions (strict format only).
    Supported actions: [A4 B2 C2] format for unit allocation
    """
    raw_action = raw_action.strip()
    
    # Boxed format extraction
    boxed_pattern = r'\\boxed{([^}]+)}'
    boxed_matches = re.findall(boxed_pattern, raw_action)
    if boxed_matches:
        # Take the last boxed content
        boxed_content = boxed_matches[-1].strip()
        
        # Check for allocation format within boxed content
        if re.match(r'^\[([A-Z]\d+\s*)+\]$', boxed_content, re.IGNORECASE):
            # Extract the content between brackets
            bracket_match = re.match(r'^\[(.+)\]$', boxed_content)
            if bracket_match:
                allocation_str = bracket_match.group(1).strip()
                # Validate that all allocations are in correct format
                allocations = re.findall(r'([A-Z])\s*(\d+)', allocation_str, re.IGNORECASE)
                if allocations:
                    # Reconstruct the allocation string
                    formatted = ' '.join([f"{field.upper()}{units}" for field, units in allocations])
                    return f"[{formatted}]", {
                        "correct_answer_format": True,
                        "extraction_method": "boxed",
                        "format_requirement_level": format_requirement_level
                    }
    
    # Bracket format extraction
    bracket_match = re.search(r'\[([^\]]+)\]', raw_action)
    if bracket_match:
        allocation_str = bracket_match.group(1).strip()
        # Extract field-unit pairs
        allocations = re.findall(r'([A-Z])\s*:?\s*(\d+)', allocation_str, re.IGNORECASE)
        if allocations:
            # Reconstruct the allocation string
            formatted = ' '.join([f"{field.upper()}{units}" for field, units in allocations])
            return f"[{formatted}]", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False,
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_debate_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse Debate actions (strict format only).
    Debate accepts free text arguments, so any non-empty text is valid.
    """
    raw_action = raw_action.strip()
    
    # For debate, any non-empty text is a valid argument
    if raw_action:
        return raw_action, {
            "correct_answer_format": True,
            "extraction_method": "free_text",
            "format_requirement_level": format_requirement_level
        }
    
    return "invalid action", {
        "correct_answer_format": False,
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


def _parse_two_dollar_action(raw_action: str, format_requirement_level: str = "strict") -> Tuple[str, Dict[str, Any]]:
    """
    Parse TwoDollar negotiation actions: [Propose] $X.XX, [Accept], [Reject].
    
    Supported formats (in order of priority):
    1. Boxed format: \\boxed{Propose $1.50}, \\boxed{Accept}, \\boxed{Reject} (strict + loose)
    2. Bracket format: [Propose] $1.50, [Accept], [Reject] (strict + loose)
    
    Validation:
    - Propose actions must include a valid dollar amount (non-negative float)
    - Accept/Reject actions require no parameters
    - Case insensitive matching but preserves proper capitalization
    
    Returns:
    - Formatted action as: "[Propose] $1.50", "[Accept]", "[Reject]"
    - Format feedback for analysis
    """
    
    # Priority 1: Try \\boxed{} format first (allowed in both strict and loose)
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', raw_action, re.IGNORECASE | re.DOTALL)
    for match in boxed_matches:
        match = match.strip()
        
        # Handle simple actions in boxed format: Accept, Reject
        if match.lower() in ["accept", "reject"]:
            action = match.capitalize()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle \\boxed{[Accept]} or \\boxed{[Reject]} format
        simple_bracket_in_boxed = re.match(r'\[(accept|reject)\]', match, re.IGNORECASE)
        if simple_bracket_in_boxed:
            action = simple_bracket_in_boxed.group(1).capitalize()
            return f"[{action}]", {
                "correct_answer_format": True,
                "extraction_method": "boxed",
                "format_requirement_level": format_requirement_level
            }
        
        # Handle propose actions in boxed format: \\boxed{Propose $1.50} or \\boxed{[Propose] $1.50}
        propose_match = re.search(r'propose\s*\$?(\d+(?:\.\d+)?)', match, re.IGNORECASE)
        if propose_match:
            amount = float(propose_match.group(1))
            if amount >= 0:
                return f"[Propose] ${amount:.2f}", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Handle \\boxed{[Propose] $1.50} format - extract from brackets
        bracket_propose_in_boxed = re.search(r'\[propose\]\s*\$(\d+(?:\.\d+)?)', match, re.IGNORECASE)
        if bracket_propose_in_boxed:
            amount = float(bracket_propose_in_boxed.group(1))
            if amount >= 0:
                return f"[Propose] ${amount:.2f}", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
        
        # Handle \\boxed{[Propose $1.50]} format - alternative bracket format
        bracket_propose_alt_in_boxed = re.search(r'\[propose\s+\$(\d+(?:\.\d+)?)\]', match, re.IGNORECASE)
        if bracket_propose_alt_in_boxed:
            amount = float(bracket_propose_alt_in_boxed.group(1))
            if amount >= 0:
                return f"[Propose] ${amount:.2f}", {
                    "correct_answer_format": True,
                    "extraction_method": "boxed",
                    "format_requirement_level": format_requirement_level
                }
    
    # Priority 2: Try bracket format [Accept], [Reject], [Propose] $X.XX (allowed in both strict and loose)
    
    # Simple actions: [Accept], [Reject]
    simple_bracket_pattern = r'\[(accept|reject)\]'
    simple_bracket_matches = re.findall(simple_bracket_pattern, raw_action, re.IGNORECASE)
    if simple_bracket_matches:
        # Take the last match
        action = simple_bracket_matches[-1].capitalize()
        return f"[{action}]", {
            "correct_answer_format": True,
            "extraction_method": "bracket",
            "format_requirement_level": format_requirement_level
        }
    
    # Propose actions: [Propose] $X.XX or [Propose $X.XX]
    propose_bracket_pattern = r'\[propose\]\s*\$(\d+(?:\.\d+)?)'
    propose_bracket_matches = re.findall(propose_bracket_pattern, raw_action, re.IGNORECASE)
    if propose_bracket_matches:
        # Take the last match
        amount = float(propose_bracket_matches[-1])
        if amount >= 0:
            return f"[Propose] ${amount:.2f}", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    # Alternative propose format: [Propose $X.XX]
    propose_bracket_alt_pattern = r'\[propose\s+\$(\d+(?:\.\d+)?)\]'
    propose_bracket_alt_matches = re.findall(propose_bracket_alt_pattern, raw_action, re.IGNORECASE)
    if propose_bracket_alt_matches:
        # Take the last match
        amount = float(propose_bracket_alt_matches[-1])
        if amount >= 0:
            return f"[Propose] ${amount:.2f}", {
                "correct_answer_format": True,
                "extraction_method": "bracket",
                "format_requirement_level": format_requirement_level
            }
    
    return "invalid action", {
        "correct_answer_format": False,
        "extraction_method": "none",
        "format_requirement_level": format_requirement_level
    }


OBSERVATION_FORMATTING: Dict[str, Callable[[str], str]] = {key: (lambda key=key: lambda observation: apply_template(key, observation))() for key in TEMPLATE_PARTS}
ACTION_EXTRACTION = {"default": lambda raw_action, env_id, format_level="strict": extract_action_and_format_feedback(raw_action, env_id, format_level)}