"""The router prompt is what actually decides whether
list_documents_mentioning ever gets called — a typo or an accidental
deletion here is invisible until someone notices cobertura questions are
still routed to `local`/`mix`. Pin the phrases the routing logic depends on.
"""

from services.agent_service import ROUTER_SKILL_CONTENT


def test_mentions_the_new_tool_by_name():
    assert "list_documents_mentioning" in ROUTER_SKILL_CONTENT


def test_step_0_precedes_the_existing_flowchart_steps():
    content = ROUTER_SKILL_CONTENT
    step0 = content.index("Step 0")
    step1 = content.index("Step 1")
    assert step0 < step1


def test_covers_both_enumeration_shapes():
    """Both "which documents mention X" (cobertura) and "list all X within
    one document" (G07-style) must be recognized in Step 0 — losing either
    regresses a whole eval-set block silently. This test extracts Step 0
    specifically to avoid false positives from Step 1-4 text."""
    content = ROUTER_SKILL_CONTENT
    # Extract Step 0 section (between "Step 0" and "Step 1")
    step0_start = content.index("Step 0")
    step1_start = content.index("Step 1")
    step0_section = content[step0_start:step1_start]

    # Both enumeration patterns must be documented with specific examples
    # These phrases are unique to Step 0 and won't appear elsewhere
    assert "In which manuals" in step0_section, \
        "Step 0 must include 'which documents mention' example (cobertura style)"
    assert "List all the parameters documented" in step0_section, \
        "Step 0 must include 'list all within document' example (G07 style)"
