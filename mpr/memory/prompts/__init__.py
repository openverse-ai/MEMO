"""
Memory system prompts and templates organized by type.
"""

# Memory merge prompts (basic style)
from .memory_merge.crud_skill_moreop_v2_prompts import (
    XML_CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE,
    XML_CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE,
)

# Abstract merge prompts (basic style)
from .abstract_merge.xml_crud_state_abstract_v2_prompts import (
    XML_CRUD_STATE_ABSTRACT_V2_PROMPT_TEMPLATE,
    XML_CRUD_STATE_ABSTRACT_V2_FORMAT_TEMPLATE,
)

# Abstract generation prompts
from .abstract_gen.basic import BASIC_ABSTRACT_GEN_PROMPT

__all__ = [
    # Memory merge
    'XML_CRUD_SKILL_MOREOP_V2_PROMPT_TEMPLATE',
    'XML_CRUD_SKILL_MOREOP_V2_FORMAT_TEMPLATE',
    # Abstract merge
    'XML_CRUD_STATE_ABSTRACT_V2_PROMPT_TEMPLATE',
    'XML_CRUD_STATE_ABSTRACT_V2_FORMAT_TEMPLATE',
    # Abstract generation
    'BASIC_ABSTRACT_GEN_PROMPT',
]
