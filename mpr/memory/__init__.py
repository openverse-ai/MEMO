"""
Memory-Enhanced Agents for Self-Play Learning.
"""

from .trajectory_memory_system import (
    TrajectoryMemorySystem,
    MemoryEnhancedAgent,
    CompressedGame,
    TrajectoryAnalyzer
)

from .prompts import (
    XML_CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE,
    XML_CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE,
    XML_CRUD_STATE_ABSTRACT_V2_PROMPT_TEMPLATE,
    XML_CRUD_STATE_ABSTRACT_V2_FORMAT_TEMPLATE,
    BASIC_ABSTRACT_GEN_PROMPT,
)

__all__ = [
    'TrajectoryMemorySystem',
    'MemoryEnhancedAgent',
    'CompressedGame',
    'TrajectoryAnalyzer',
    'XML_CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE',
    'XML_CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE',
    'XML_CRUD_STATE_ABSTRACT_V2_PROMPT_TEMPLATE',
    'XML_CRUD_STATE_ABSTRACT_V2_FORMAT_TEMPLATE',
    'BASIC_ABSTRACT_GEN_PROMPT',
]
