"""Per-App fact-extraction cadence (spec `user-memory-mem0`, D-15):
`App.memory_write_every_n_turns` overrides the deployment default
(`AICT_GLOBAL_MEMORY_WRITE_EVERY_N_TURNS`); NULL or an invalid value falls
back to it. Extraction always runs on turn 1 and then on every Nth turn.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import config
from repositories.app_repository import AppRepository
from schemas.apps_schemas import UpdateAppSchema
from tools.memory import settings as memory_settings

from tests.unit.services.test_agent_execution_service_write_scheduling import (
    _finalize,
    _identity,
    _make_context,
    _make_service,
)


class TestEffectiveCadence:
    def test_the_apps_value_wins(self):
        assert memory_settings.write_every_n_turns(SimpleNamespace(memory_write_every_n_turns=3)) == 3

    @pytest.mark.parametrize("value", [None, 0, -2, True, "3"])
    def test_missing_or_invalid_values_fall_back_to_the_deployment_default(self, value):
        app = SimpleNamespace(memory_write_every_n_turns=value)
        assert memory_settings.write_every_n_turns(app) == config.GLOBAL_MEMORY_WRITE_EVERY_N_TURNS

    def test_no_app_uses_the_deployment_default(self):
        assert memory_settings.write_every_n_turns(None) == config.GLOBAL_MEMORY_WRITE_EVERY_N_TURNS

    @pytest.mark.parametrize("raw, expected", [("7", 7), ("1", 1), ("0", 5), ("-3", 5), ("abc", 5), ("", 5)])
    def test_the_environment_value_is_validated(self, raw, expected):
        """`turn % 0` would raise inside the write hook and silently stop all
        writes, so anything below 1 falls back to 5."""
        assert config._parse_write_every_n_turns(raw) == expected


class TestTrigger:
    def test_turn_1_and_every_nth_turn(self):
        fired = [t for t in range(1, 13) if memory_settings.should_write(t, 3)]
        assert fired == [1, 3, 6, 9, 12]

    def test_every_turn_when_n_is_1(self):
        assert all(memory_settings.should_write(t, 1) for t in range(1, 6))

    def test_omitted_cadence_uses_the_default(self):
        default = config.GLOBAL_MEMORY_WRITE_EVERY_N_TURNS
        assert memory_settings.should_write(default) is True
        assert memory_settings.should_write(default + 1) is (default == 1)


class TestApiSurface:
    @pytest.mark.parametrize("value", [1, 5, 100, None])
    def test_valid_values_are_accepted(self, value):
        assert UpdateAppSchema(name="App", memory_write_every_n_turns=value).memory_write_every_n_turns == value

    @pytest.mark.parametrize("value", [0, -1, 101])
    def test_out_of_range_values_are_rejected(self, value):
        with pytest.raises(ValidationError):
            UpdateAppSchema(name="App", memory_write_every_n_turns=value)

    def test_omitting_the_field_leaves_it_unset(self):
        """The PUT handler only forwards it when present in `model_fields_set`."""
        assert "memory_write_every_n_turns" not in UpdateAppSchema(name="App").model_fields_set

    def test_the_repository_persists_it_only_when_given(self):
        app = SimpleNamespace(memory_write_every_n_turns=None)
        repo = AppRepository(MagicMock())

        repo._update_app_fields(app, {"name": "App", "memory_write_every_n_turns": 3})
        assert app.memory_write_every_n_turns == 3

        repo._update_app_fields(app, {"name": "App"})
        assert app.memory_write_every_n_turns == 3


class TestFinalizeTurnUsesTheAppsCadence:
    def _ctx(self, turn, every_n):
        app = SimpleNamespace(memory_write_every_n_turns=every_n)
        fresh_agent = MagicMock(ai_service=None, has_global_memory=True, app=app)
        conversation = MagicMock(conversation_id=1, user_id=7, memory_pending_user_messages=None)
        return _make_context(
            global_memory_identity=_identity(), memory_turn_index=turn,
            fresh_agent=fresh_agent, conversation=conversation,
            original_user_message=f"turn {turn}",
        )

    @pytest.mark.asyncio
    async def test_writes_on_the_apps_cadence(self):
        svc = _make_service()
        fired = []
        for turn in range(1, 8):
            with (
                patch.object(memory_settings, "is_global_memory_enabled", return_value=True),
                patch("services.conversation_service.ConversationService.increment_message_count"),
                patch("services.global_memory_service.take_pending_user_messages", return_value=[]),
                patch("services.global_memory_service.buffer_pending_user_message"),
                patch("services.global_memory_service.schedule_write_path") as mock_schedule,
            ):
                await _finalize(svc, self._ctx(turn, every_n=3))
            if mock_schedule.called:
                fired.append(turn)

        assert fired == [1, 3, 6]

    @pytest.mark.asyncio
    async def test_the_pending_buffer_is_sized_to_the_apps_cadence(self):
        svc = _make_service()
        with (
            patch.object(memory_settings, "is_global_memory_enabled", return_value=True),
            patch("services.conversation_service.ConversationService.increment_message_count"),
            patch("services.global_memory_service.buffer_pending_user_message") as mock_buffer,
            patch("services.global_memory_service.schedule_write_path"),
        ):
            await _finalize(svc, self._ctx(turn=2, every_n=3))

        assert mock_buffer.call_args.args[3] == 3
