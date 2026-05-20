"""Unit tests for ``tools.agentTools.IACTTool`` (agent-as-tool wrapper).

Focus: the async factory ``IACTTool.create`` must load the sub-agent's MCP tools
(the bug being fixed), and must keep working when the MCP server fails.

All external dependencies are mocked — no LLM, MCP server or database is touched.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.agent import Agent
from tools import agentTools


def _make_agent(name: str = "Sub Agent") -> Agent:
    """Build a transient (un-persisted) Agent suitable for IACTTool construction."""
    return Agent(name=name, description=f"{name} description", system_prompt="")


@pytest.mark.asyncio
async def test_iact_tool_create_builds_react_agent():
    """The async factory returns a ready IACTTool with its react_agent built."""
    agent = _make_agent()

    with (
        patch("tools.agentTools.get_llm", return_value=object()),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()),
        patch.object(agentTools.MCPClientManager, "get_client", new=AsyncMock(return_value=None)),
    ):
        tool = await agentTools.IACTTool.create(agent)

    assert isinstance(tool, agentTools.IACTTool)
    assert tool.react_agent is not None


@pytest.mark.asyncio
async def test_iact_tool_create_loads_sub_agent_mcp_tools():
    """A sub-agent's MCP tools are loaded and added to its react_agent."""
    agent = _make_agent("MCP Sub-Agent")

    fake_mcp_tool = MagicMock()
    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[fake_mcp_tool])

    with (
        patch("tools.agentTools.get_llm", return_value=object()),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as mock_create,
        patch.object(agentTools.MCPClientManager, "get_client", new=AsyncMock(return_value=fake_client)),
    ):
        tool = await agentTools.IACTTool.create(agent, user_context={"user_id": 1})

    loaded_tools = mock_create.call_args.kwargs["tools"]
    assert fake_mcp_tool in loaded_tools
    # The MCP client is kept on the instance so its connection lives as long as the tool.
    assert tool.mcp_client is fake_client


@pytest.mark.asyncio
async def test_iact_tool_create_survives_mcp_load_failure():
    """A failing MCP server degrades the sub-agent but never breaks construction."""
    agent = _make_agent("MCP Sub-Agent")

    with (
        patch("tools.agentTools.get_llm", return_value=object()),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as mock_create,
        patch.object(
            agentTools.MCPClientManager,
            "get_client",
            new=AsyncMock(side_effect=RuntimeError("MCP server down")),
        ),
    ):
        tool = await agentTools.IACTTool.create(agent)

    assert tool.react_agent is not None
    assert tool.mcp_client is None
    # Base tools are still present despite the MCP failure.
    assert agentTools.get_current_date in mock_create.call_args.kwargs["tools"]
