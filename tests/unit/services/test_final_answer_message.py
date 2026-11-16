"""_final_answer_message: never hand a raw tool dump back as the model's answer."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.services.agent_execution_service import _final_answer_message as pick


def _tool(text="RAW CHUNKS"):
    return ToolMessage(content=text, tool_call_id="t1")


def test_normal_ai_answer():
    msgs = [HumanMessage("q"), AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "t1"}]), _tool(), AIMessage("answer")]
    assert pick(msgs).content == "answer"


def test_empty_final_ai_does_not_fall_back_to_tool_output():
    msgs = [HumanMessage("q"), AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "t1"}]), _tool(), AIMessage("")]
    assert pick(msgs) is None


def test_return_direct_turn_ends_on_tool_message():
    msgs = [HumanMessage("q"), AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "t1"}]), _tool("specialist answer")]
    assert pick(msgs).content == "specialist answer"
