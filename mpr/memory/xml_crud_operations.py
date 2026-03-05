"""
Simple tag-based CRUD operations for memory management.
Uses regex parsing with simple XML-like tags for reliable operation extraction.
Format: <add>content</add>, <edit number="3">content</edit>, <remove number="5">reason</remove>
"""

from typing import List, Tuple, Optional


class XMLCRUDParser:
    """Parse and apply CRUD operations using simple tag-based format with regex."""
    
    def parse_operations(self, xml_response: str) -> List[Tuple[str, str, Optional[int]]]:
        """
        Parse operations from LLM response using simple regex patterns.
        
        Expected format:
        <add>New skill content here</add>
        <edit number="3">Updated skill content here</edit>
        <remove number="5">Optional reason for removal</remove>
        
        Returns:
            List of tuples: (operation_type, text, insight_number)
            - operation_type: 'ADD', 'EDIT', or 'REMOVE'
            - text: The insight text (or reason for REMOVE)
            - insight_number: The insight number for EDIT/REMOVE (None for ADD)
        """
        operations = []
        import re
        
        # Simple regex patterns for direct tag matching
        add_pattern = r'<add>(.*?)</add>'
        edit_pattern = r'<edit\s+number="(\d+)">(.*?)</edit>'
        remove_pattern = r'<remove\s+number="(\d+)">(.*?)</remove>'
        
        # Find ADD operations
        for match in re.finditer(add_pattern, xml_response, re.DOTALL):
            content = match.group(1).strip()
            if content:
                operations.append(('ADD', content, None))
        
        # Find EDIT operations
        for match in re.finditer(edit_pattern, xml_response, re.DOTALL):
            number = int(match.group(1))  # Will raise ValueError if invalid
            content = match.group(2).strip()
            if content:
                operations.append(('EDIT', content, number))
        
        # Find REMOVE operations
        for match in re.finditer(remove_pattern, xml_response, re.DOTALL):
            number = int(match.group(1))  # Will raise ValueError if invalid
            reason = match.group(2).strip()
            operations.append(('REMOVE', reason, number))
        
        return operations
    

    def apply_operations(
        self, 
        existing_insights: List[str], 
        operations: List[Tuple[str, str, Optional[int]]],
        max_insights: int = 15
    ) -> List[str]:
        """
        Apply parsed operations to existing insights.
        
        Args:
            existing_insights: Current list of insights
            operations: List of (operation_type, text, insight_number) tuples
            max_insights: Maximum number of insights to keep
            
        Returns:
            Updated list of insights
        """
        # Start with a copy
        updated_insights = list(existing_insights)
        
        # Track which indices to remove (process in reverse order later)
        indices_to_remove = []
        
        # Process operations by type for better control
        # First: REMOVE operations
        for op_type, text, number in operations:
            if op_type == 'REMOVE' and number is not None:
                # Convert to 0-based index
                idx = number - 1
                if 0 <= idx < len(updated_insights):
                    indices_to_remove.append(idx)
        
        # Apply removals in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            del updated_insights[idx]
        
        # Second: EDIT operations
        for op_type, text, number in operations:
            if op_type == 'EDIT' and number is not None and text:
                # Convert to 0-based index
                idx = number - 1
                # Adjust index if removals were applied
                adjusted_idx = idx
                for removed_idx in sorted(indices_to_remove):
                    if removed_idx < idx:
                        adjusted_idx -= 1
                
                if 0 <= adjusted_idx < len(updated_insights):
                    updated_insights[adjusted_idx] = text
        
        # Third: ADD operations
        for op_type, text, number in operations:
            if op_type == 'ADD' and text:
                # Check for duplicates
                is_duplicate = False
                for existing in updated_insights:
                    # Simple similarity check
                    existing_words = set(existing.lower().split())
                    new_words = set(text.lower().split())
                    overlap = len(existing_words & new_words)
                    if overlap > min(len(existing_words), len(new_words)) * 0.6:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    updated_insights.append(text)
        
        # Limit total insights
        if len(updated_insights) > max_insights:
            updated_insights = updated_insights[-max_insights:]
        
        # Ensure we have at least some insights
        if not updated_insights and existing_insights:
            return existing_insights
        
        return updated_insights


def create_xml_crud_example() -> str:
    """Create an example showing simple tag-based CRUD operations."""
    return """Example of using simple tag-based CRUD operations:

Given these existing insights:
1. Control center positions early to establish dominance
2. Always bet aggressively with strong hands  
3. Check action format before submitting

The LLM response might contain:

<add>Establish early positional advantage through strategic piece placement and territory control.</add>
<edit number="2">Balance aggressive plays with conservative resource management to maintain strategic flexibility.</edit>
<remove number="1">Too specific to one game type</remove>
<add>Adapt decision-making frameworks based on opponent behavioral patterns and game state evolution.</add>

Processing steps:
1. REMOVE insight #1 (removes first item)
2. EDIT insight #2 (updates second item, now at index 1)  
3. ADD two new insights

Final result:
1. Balance aggressive plays with conservative resource management to maintain strategic flexibility. (edited #2)
2. Check action format before submitting (original #3)
3. Establish early positional advantage through strategic piece placement and territory control. (new)
4. Adapt decision-making frameworks based on opponent behavioral patterns and game state evolution. (new)

Note: Uses simple regex patterns to find <add>, <edit>, and <remove> tags - no XML parser needed."""