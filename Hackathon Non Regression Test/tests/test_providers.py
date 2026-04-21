"""Test per src.providers."""

from unittest.mock import MagicMock, patch

from src.providers.bedrock import BedrockClient


class TestBedrockClient:
    def test_init_creates_client(self):
        with patch("src.providers.bedrock.boto3.client") as mock_boto:
            client = BedrockClient(region="eu-west-1")
            mock_boto.assert_called_once()
            call_kwargs = mock_boto.call_args
            assert call_kwargs[0][0] == "bedrock-runtime"
            assert call_kwargs[1]["region_name"] == "eu-west-1"
            assert client.region == "eu-west-1"

    def test_invoke_returns_expected_keys(self):
        with patch("src.providers.bedrock.boto3.client") as mock_boto:
            mock_runtime = MagicMock()
            mock_boto.return_value = mock_runtime
            mock_runtime.converse.return_value = {
                "output": {
                    "message": {
                        "content": [{"text": "Hello world"}]
                    }
                },
                "usage": {"inputTokens": 10, "outputTokens": 5},
                "stopReason": "end_turn",
            }

            client = BedrockClient(region="eu-west-1")
            result = client.invoke(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

            assert result["output_text"] == "Hello world"
            assert result["input_tokens"] == 10
            assert result["output_tokens"] == 5
            assert result["stop_reason"] == "end_turn"

    def test_invoke_extracts_system_from_messages(self):
        with patch("src.providers.bedrock.boto3.client") as mock_boto:
            mock_runtime = MagicMock()
            mock_boto.return_value = mock_runtime
            mock_runtime.converse.return_value = {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 5, "outputTokens": 2},
                "stopReason": "end_turn",
            }

            client = BedrockClient()
            client.invoke(
                model_id="test-model",
                messages=[
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                ],
            )

            call_kwargs = mock_runtime.converse.call_args[1]
            assert "system" in call_kwargs
            assert call_kwargs["system"] == [{"text": "Be helpful."}]
            # System message should NOT be in messages
            roles = [m["role"] for m in call_kwargs["messages"]]
            assert "system" not in roles
