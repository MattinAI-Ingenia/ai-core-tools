"""Editing an AI service from the form must not wipe extra_config keys it does not manage."""
import json

from services.ai_service_service import AIServiceService


def test_edit_keeps_model_params_and_replaces_bedrock_fields():
    existing = json.dumps({"top_p": 0.95, "reasoning_effort": "max", "aws_region": "us-east-1"})
    merged = json.loads(AIServiceService._merge_extra_config(existing, json.dumps({"aws_region": "eu-west-1"})))
    assert merged == {"top_p": 0.95, "reasoning_effort": "max", "aws_region": "eu-west-1"}


def test_edit_without_anything_left_stays_null():
    assert AIServiceService._merge_extra_config(None, None) is None
    assert AIServiceService._merge_extra_config('{"aws_region": "x"}', None) is None


def test_edit_tolerates_broken_existing_json():
    assert json.loads(AIServiceService._merge_extra_config("{nope", '{"aws_region": "eu"}')) == {"aws_region": "eu"}
