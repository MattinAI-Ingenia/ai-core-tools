"""get_valid_tool_ids must only accept agents of the caller's own App.

The UI only lists same-app tool agents, but the API trusted whatever ids it
got: a crafted request could attach another App's agent as a tool.
"""
from models.agent import Agent
from models.app import App
from repositories.agent_repository import AgentRepository


def _tool_agent(db, app_id, name):
    agent = Agent(name=name, description="", system_prompt="", app_id=app_id, is_tool=True)
    db.add(agent)
    db.flush()
    return agent


def test_rejects_tool_agents_from_another_app(db, fake_app, fake_user):
    other_app = App(
        name="Other Workspace", slug="other-workspace-tool-ids", owner_id=fake_user.user_id,
        agent_rate_limit=0, max_file_size_mb=10,
    )
    db.add(other_app)
    db.flush()
    own = _tool_agent(db, fake_app.app_id, "own")
    foreign = _tool_agent(db, other_app.app_id, "foreign")

    valid = AgentRepository.get_valid_tool_ids(db, [own.agent_id, foreign.agent_id], fake_app.app_id)

    assert valid == [own.agent_id]
