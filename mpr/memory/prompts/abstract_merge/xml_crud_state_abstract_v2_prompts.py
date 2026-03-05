"""
Enhanced XML CRUD State Abstract memory merge prompts focused on diverse strategic state pattern recognition.
Improvements based on analysis:
- Prioritize diversity in board states over redundant analyses
- Maintain specific actionable advice with positions/cells
- Encourage comprehensive coverage of different game scenarios
- Prevent accumulation of similar state analyses
"""

XML_CRUD_STATE_ABSTRACT_V2_FORMAT_TEMPLATE = """OPERATION FORMAT:
Use simple XML tags for each operation:

<add>New state analysis with strategic pattern examples.</add>
<edit number="3">Updated state analysis with improved strategic insights.</edit>
<remove number="5">Why this state analysis should be removed</remove>

OPERATION GUIDELINES:
- ADD: For new state analyses covering unique board configurations or strategic scenarios
- EDIT: To merge similar states or enhance existing analyses with more specific advice
- REMOVE: For redundant states, duplicate board patterns, or analyses lacking actionable guidance

QUALITY REQUIREMENTS:
- Include SPECIFIC positions, cells, or moves (e.g., "cell 3", "position 5")
- Provide actionable advice addressing the state's win/loss variance
- Balance offensive opportunities with defensive necessities
- Help players convert losses into wins or draws
- Prioritize diverse board states over duplicate analyses

TECHNICAL REQUIREMENTS:
- Use the 'number' attribute for EDIT/REMOVE operations (1-based numbering)
- If library is empty, use ONLY ADD operations
- Never reference non-existent state analysis numbers

Example operations:
<add>STATE: {{"board": [["X", "O", ""], ["", "X", ""], ["", "", "O"]]}}
ABSTRACT: This early diagonal formation is critical because X controls the center while O has corner positions. Players should immediately block cell 8 to prevent O from completing the diagonal, while X should consider cell 6 to create dual threats. The 15 wins vs 8 losses from this state show that controlling both diagonals early provides significant advantage.</add>
<edit number="2">STATE: {{"board": [["X", "X", "O"], ["O", "O", "X"], ["", "", ""]]}}
ABSTRACT: This late-game state with an open bottom row is decisive (0 wins, 36 losses) because the player failed to block imminent threats. The critical move is placing in cell 7 to prevent the opponent's horizontal win, while also considering cell 8 to create defensive flexibility. This state demonstrates the importance of threat assessment over offensive positioning in constrained endgames.</edit>
<remove number="7">Redundant analysis - already covered by state analysis #3 with identical board configuration</remove>"""

XML_CRUD_STATE_ABSTRACT_V2_PROMPT_TEMPLATE = """You are maintaining a state analysis library for strategic game pattern recognition. Update the library by performing operations on the state analyses.

NEW STATE ANALYSES FROM RECENT GAMES:
{new_abstracts_formatted}

EXISTING STATE ANALYSIS LIBRARY:
{old_abstracts_formatted}

{format_template}

MERGE APPROACH:
1. Identify new analyses covering unique board states not in the library
2. Consolidate similar board positions through EDIT or REMOVE operations
3. Ensure the library represents diverse game phases (opening, midgame, endgame)

Generate your operations below:
"""
