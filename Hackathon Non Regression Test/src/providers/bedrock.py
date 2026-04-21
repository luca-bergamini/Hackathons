"""Client wrapper per Amazon Bedrock."""

import logging
import os

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# Alza il pool di connessioni HTTP per supportare alto parallelismo (default boto3 = 10).
_MAX_POOL = int(os.environ.get("EVAL_MAX_WORKERS", 50)) + 5


class BedrockClient:
    """Wrapper per invocare qualsiasi modello su Amazon Bedrock."""

    def __init__(self, region: str = "eu-west-1"):
        self.region = region
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(max_pool_connections=_MAX_POOL),
        )

    def invoke(
        self,
        model_id: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        """Ritorna un dict con almeno: output_text, input_tokens, output_tokens."""
        # Build converse-format messages: [{"role": ..., "content": [{"text": ...}]}]
        converse_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            # Skip system messages — handled separately via system kwarg
            if role == "system":
                if system is None:
                    system = msg.get("content", "")
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"text": content}]
            converse_messages.append({"role": role, "content": content})

        # Bedrock Converse API requires messages to start with role "user".
        # If the first message is not "user" (e.g. assistant), or list is empty,
        # prepend a minimal user message to satisfy the API constraint.
        if not converse_messages or converse_messages[0]["role"] != "user":
            converse_messages.insert(0, {"role": "user", "content": [{"text": "Please proceed."}]})

        kwargs = {
            "modelId": model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self._client.converse(**kwargs)

        output_message = response.get("output", {}).get("message", {})
        output_text = ""
        for block in output_message.get("content", []):
            if "text" in block:
                output_text += block["text"]

        usage = response.get("usage", {})

        return {
            "output_text": output_text,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "stop_reason": response.get("stopReason", ""),
        }
