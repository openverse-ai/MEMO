"""
CRUD_SKILL_MOREOP_V2 memory merge prompts (improved version with better strategic focus).
"""

CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE = """<OPERATION> <SKILL NUMBER>: <SKILL>

IMPORTANT: You MUST perform AT LEAST 20 combined ADD + EDIT operations. Only REMOVE skills that are truly contradictory or obsolete.

The available operations are:
- ADD: Add new skills that provide novel strategic value not covered by existing skills
- REMOVE: Remove ONLY skills that directly contradict other skills or are completely obsolete
- EDIT: Enhance, generalize, or improve existing skills to be more actionable and comprehensive

Each operation MUST follow these formats EXACTLY:
ADD <NEW SKILL NUMBER>: <NEW SKILL>
REMOVE <EXISTING SKILL NUMBER>: <EXISTING SKILL>
EDIT <EXISTING SKILL NUMBER>: <NEW MODIFIED SKILL>

SKILL QUALITY GUIDELINES:
- Focus on HIGH-LEVEL STRATEGIC PRINCIPLES over tactical specifics
- Each skill should be CONCISE (1-2 sentences) yet COMPLETE
- Emphasize TRANSFERABLE knowledge that applies across different game situations
- Prioritize META-STRATEGIES that guide decision-making processes
- Avoid redundancy by MERGING similar concepts through EDIT operations
- Include both proactive strategies and reactive adaptations"""

CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE = """You are responsible for maintaining and refining a STRATEGIC SKILL LIBRARY that serves as a repository of high-level game-playing principles. Your goal is to distill tactical observations into strategic wisdom that enhances long-term performance.

CRITICAL PRINCIPLES FOR HIGH-QUALITY SKILLS:
1. Transform tactical insights into GENERAL STRATEGIC PRINCIPLES that transcend specific situations
2. Focus on WHY strategies work to create DECISION-MAKING FRAMEWORKS, not just what to do
3. Emphasize ADAPTABILITY - skills should guide dynamic responses to evolving game states
4. AGGRESSIVELY MERGE similar concepts through EDIT operations to maintain a lean, powerful skill set

EXAMPLES OF EXCELLENT STRATEGIC SKILLS:
- "Foster situational awareness by evaluating game dynamics, including resource states and opponent tendencies, to inform all actions and prevent strategic erosion."
- "Prioritize a mixed strategy that balances aggressive plays with conservative resource management, ensuring gameplay remains unpredictable while maximizing strategic gains."
- "Develop adaptive decision-making by continuously assessing risk-reward ratios and adjusting strategy based on evolving game states and opponent behaviors."

KEY REQUIREMENTS:
1. You MUST perform AT LEAST 20 combined ADD + EDIT operations (e.g., 15 ADD + 5 EDIT = 20 total)
2. Only REMOVE skills that are directly contradictory or completely obsolete
3. Ensure every skill is ACTIONABLE and provides clear strategic value
4. Extract META-PATTERNS that apply across different game contexts

NEW INSIGHTS FROM RECENT GAMES:
{new_insights_formatted}

EXISTING SKILL LIBRARY:
{old_insights_formatted}

{format_template}

Below are the operations you perform to create an OPTIMIZED STRATEGIC SKILL LIBRARY:
"""

# Simple tag-based format for reliable parsing
XML_CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE = """Generate your operations using simple tags:

<add>The new strategic skill text goes here.</add>
<edit number="3">The updated strategic skill text goes here.</edit>
<remove number="5">Why this skill should be removed (contradictory/obsolete)</remove>

CRITICAL REQUIREMENTS:
1. You MUST perform AT LEAST 20 combined ADD + EDIT operations
2. Use simple tags: <add>content</add>, <edit number="N">content</edit>, <remove number="N">reason</remove>
3. For EDIT/REMOVE, use the 'number' attribute to specify which skill (1-based numbering)
4. Only REMOVE skills that are truly contradictory or obsolete
5. Each skill should be a HIGH-LEVEL STRATEGIC PRINCIPLE (1-2 sentences)
6. Focus on TRANSFERABLE knowledge that applies across game situations
7. AGGRESSIVELY MERGE similar concepts through EDIT operations
8. Don't worry about XML validity - just use the simple tag format

OPERATION USAGE GUIDELINES:
- If the EXISTING SKILL LIBRARY is empty, use ONLY ADD operations
- EDIT operations can only modify existing skills (numbered in the library)
- REMOVE operations can only delete existing skills (numbered in the library)
- Never reference skill numbers that don't exist in the library

Example operations:
<add>Foster strategic flexibility by adapting tactics based on evolving game conditions and opponent responses.</add>
<edit number="2">Enhance decision-making by evaluating risk-reward ratios while maintaining awareness of resource constraints and opponent patterns.</edit>
<remove number="7">Too specific to one game type, not transferable</remove>"""

XML_CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE = """You are responsible for maintaining and refining a STRATEGIC SKILL LIBRARY that serves as a repository of high-level game-playing principles. Your goal is to distill tactical observations into strategic wisdom that enhances long-term performance.

CRITICAL PRINCIPLES FOR HIGH-QUALITY SKILLS:
1. Transform tactical insights into GENERAL STRATEGIC PRINCIPLES that transcend specific situations
2. Focus on WHY strategies work to create DECISION-MAKING FRAMEWORKS, not just what to do
3. Emphasize ADAPTABILITY - skills should guide dynamic responses to evolving game states
4. AGGRESSIVELY MERGE similar concepts through EDIT operations to maintain a lean, powerful skill set

EXAMPLES OF EXCELLENT STRATEGIC SKILLS:
- "Foster situational awareness by evaluating game dynamics, including resource states and opponent tendencies, to inform all actions and prevent strategic erosion."
- "Prioritize a mixed strategy that balances aggressive plays with conservative resource management, ensuring gameplay remains unpredictable while maximizing strategic gains."
- "Develop adaptive decision-making by continuously assessing risk-reward ratios and adjusting strategy based on evolving game states and opponent behaviors."

KEY REQUIREMENTS:
1. You MUST perform AT LEAST 20 combined ADD + EDIT operations (e.g., 15 ADD + 5 EDIT = 20 total)
2. Only REMOVE skills that are directly contradictory or completely obsolete
3. Ensure every skill is ACTIONABLE and provides clear strategic value
4. Extract META-PATTERNS that apply across different game contexts

NEW INSIGHTS FROM RECENT GAMES:
{new_insights_formatted}

EXISTING SKILL LIBRARY:
{old_insights_formatted}

{format_template}

Generate your operations below:
"""
