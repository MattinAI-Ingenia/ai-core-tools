"""Erasure also drops buffered pending user messages (spec `user-memory-mem0`,
D-15): otherwise a fact the user just erased would be re-learned at the next
write-due turn from messages they sent before erasing it.

Runs against the (savepoint-isolated) test database so the scoping query —
by App through the conversation's agent, by user, or both — is exercised for
real.
"""
import json
import uuid
from unittest.mock import patch

from models.conversation import Conversation
from services.global_memory_service import _clear_pending_best_effort, delete_global_memory
from tools.memory.client import PurgeResult

from tests.factories import AgentFactory, AppFactory, UserFactory, configure_factories


def _conversation(db, agent, user):
    conversation = Conversation(
        agent_id=agent.agent_id,
        user_id=user.user_id,
        session_id=f"conv_{uuid.uuid4()}",
        memory_pending_user_messages=json.dumps(["contéstame siempre en inglés"]),
    )
    db.add(conversation)
    db.flush()
    return conversation


def _pending(db, conversation):
    db.refresh(conversation)
    return conversation.memory_pending_user_messages


class TestClearPendingScoping:
    def test_app_and_user_scope_clears_only_that_users_conversations_in_that_app(self, db):
        configure_factories(db)
        user, other_user = UserFactory(), UserFactory()
        app_a, app_b = AppFactory(owner=user), AppFactory(owner=user)
        agent_a, agent_b = AgentFactory(app=app_a), AgentFactory(app=app_b)
        target = _conversation(db, agent_a, user)
        same_user_other_app = _conversation(db, agent_b, user)
        other_user_same_app = _conversation(db, agent_a, other_user)

        _clear_pending_best_effort(db, app_id=app_a.app_id, user_id=user.user_id)

        assert _pending(db, target) is None
        assert _pending(db, same_user_other_app) is not None
        assert _pending(db, other_user_same_app) is not None

    def test_user_scope_clears_that_user_in_every_app(self, db):
        configure_factories(db)
        user, other_user = UserFactory(), UserFactory()
        app_a, app_b = AppFactory(owner=user), AppFactory(owner=user)
        in_a = _conversation(db, AgentFactory(app=app_a), user)
        in_b = _conversation(db, AgentFactory(app=app_b), user)
        someone_else = _conversation(db, AgentFactory(app=app_a), other_user)

        _clear_pending_best_effort(db, user_id=user.user_id)

        assert _pending(db, in_a) is None
        assert _pending(db, in_b) is None
        assert _pending(db, someone_else) is not None

    def test_a_failure_never_raises_and_leaves_the_callers_transaction_usable(self, db):
        """Reached mid-cascade from `UserService.delete_user`: a failure must
        roll back only its own SAVEPOINT."""
        configure_factories(db)
        user = UserFactory()
        db.flush()

        with patch(
            "services.global_memory_service._clear_pending_user_messages", side_effect=RuntimeError("boom")
        ):
            _clear_pending_best_effort(db, user_id=user.user_id)

        db.refresh(user)  # the session is still usable
        assert user.user_id is not None


class TestSelfServiceErasureClearsPending:
    def test_delete_global_memory_clears_the_scopes_pending_messages(self, db):
        configure_factories(db)
        user = UserFactory()
        app_obj = AppFactory(owner=user)
        conversation = _conversation(db, AgentFactory(app=app_obj), user)

        with patch("services.global_memory_service.purge", return_value=PurgeResult(deleted=1, complete=True)):
            delete_global_memory(db, app_obj.app_id, user.user_id)

        assert _pending(db, conversation) is None
