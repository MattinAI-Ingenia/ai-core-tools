"""Tests for azure_endpoint module."""
import pytest
from utils.azure_endpoint import normalize_azure_openai_endpoint


class TestNormalizeAzureOpenaiEndpoint:
    """Tests for normalize_azure_openai_endpoint."""

    def test_bare_resource_endpoint_gets_deployment_path(self):
        assert (
            normalize_azure_openai_endpoint("https://x.openai.azure.com", "gpt-4o")
            == "https://x.openai.azure.com/openai/deployments/gpt-4o"
        )

    def test_trailing_slash_is_stripped_then_appended(self):
        assert (
            normalize_azure_openai_endpoint("https://x.openai.azure.com/", "gpt-4o")
            == "https://x.openai.azure.com/openai/deployments/gpt-4o"
        )

    def test_full_deployment_path_unchanged(self):
        endpoint = "https://x.openai.azure.com/openai/deployments/gpt-5.4-mini"
        assert normalize_azure_openai_endpoint(endpoint, "gpt-4o") == endpoint

    def test_openai_v1_path_unchanged(self):
        endpoint = "https://x.openai.azure.com/openai/v1"
        assert normalize_azure_openai_endpoint(endpoint, "gpt-4o") == endpoint

    def test_non_azure_openai_host_unchanged(self):
        endpoint = "https://my-resource.services.ai.azure.com/models"
        assert normalize_azure_openai_endpoint(endpoint, "gpt-4o") == endpoint

    def test_custom_gateway_unchanged(self):
        endpoint = "https://internal-gateway.example.com/azure"
        assert normalize_azure_openai_endpoint(endpoint, "gpt-4o") == endpoint

    def test_empty_endpoint_returns_empty(self):
        assert normalize_azure_openai_endpoint("", "gpt-4o") == ""

    def test_none_endpoint_returns_none(self):
        assert normalize_azure_openai_endpoint(None, "gpt-4o") is None

    def test_bare_openai_path_word_is_untouched(self):
        endpoint = "https://x.openai.azure.com/openai"
        assert normalize_azure_openai_endpoint(endpoint, "gpt-4o") == endpoint

    def test_openai_subpath_word_is_untouched(self):
        endpoint = "https://x.openai.azure.com/openaifoo"
        assert normalize_azure_openai_endpoint(endpoint, "gpt-4o") == endpoint

    def test_whitespace_is_stripped(self):
        assert (
            normalize_azure_openai_endpoint("  https://x.openai.azure.com  ", "gpt-4o")
            == "https://x.openai.azure.com/openai/deployments/gpt-4o"
        )

    def test_hostname_check_is_case_insensitive(self):
        assert (
            normalize_azure_openai_endpoint("https://X.OpenAI.Azure.Com", "gpt-4o")
            == "https://X.OpenAI.Azure.Com/openai/deployments/gpt-4o"
        )

    def test_existing_query_and_fragment_survive_the_expansion(self):
        assert (
            normalize_azure_openai_endpoint("https://x.openai.azure.com?keep=1", "gpt-4o")
            == "https://x.openai.azure.com/openai/deployments/gpt-4o?keep=1"
        )
        assert (
            normalize_azure_openai_endpoint("https://x.openai.azure.com#anchor", "gpt-4o")
            == "https://x.openai.azure.com/openai/deployments/gpt-4o#anchor"
        )

    def test_url_special_characters_in_the_model_are_percent_encoded(self):
        assert (
            normalize_azure_openai_endpoint("https://x.openai.azure.com", "gpt 4o?/?#")
            == "https://x.openai.azure.com/openai/deployments/gpt%204o%3F%2F%3F%23"
        )

    def test_empty_model_returns_the_url_unchanged(self):
        assert (
            normalize_azure_openai_endpoint("https://x.openai.azure.com", "")
            == "https://x.openai.azure.com"
        )

    def test_trailing_slash_path_joins_cleanly(self):
        assert (
            normalize_azure_openai_endpoint("https://x.openai.azure.com//", "gpt-4o")
            == "https://x.openai.azure.com/openai/deployments/gpt-4o"
        )
