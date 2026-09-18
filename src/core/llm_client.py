"""
Shared LLM client for ApplyCopilot.

Centralizes OpenCode Go client construction (single source for base URL,
auth, session headers) and the chat-completion call pattern used by the
cover letter generator and the employer Q&A chatbot.

OpenCode Go requires a stable session ID in the `x-opencode-session` header
per conversation (enforced since Sep 2026) and expects clients to identify
themselves with their own User-Agent.
See https://opencode.ai/docs/go/#where-can-i-use-it
"""

import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI

from src.config.logging_config import setup_logger
from src.config.settings import LLM_MODEL

load_dotenv()
logger = setup_logger(__name__)

# OpenCode Go endpoint settings
LLM_BASE_URL = "https://opencode.ai/zen/go/v1"
LLM_USER_AGENT = "apply-copilot/1.0"
LLM_TEMPERATURE = 0.6


class LLMClient:
    """OpenCode Go LLM client with a stable per-conversation session ID."""

    def __init__(self, llm_model: str = LLM_MODEL):
        """
        Initialize the LLM client.

        Args:
            llm_model: LLM model name (OpenCode OpenAI-compatible endpoint)
        """
        self.model = llm_model
        self.session_id = str(uuid.uuid4())
        self.client = self._create_openai_client()
        logger.info(f"Initialized LLMClient with model: {llm_model}")

    def _create_openai_client(self) -> OpenAI:
        """Create the OpenAI-compatible client pointed at the OpenCode Go endpoint."""
        return OpenAI(
            base_url=LLM_BASE_URL,
            api_key=os.getenv("OPENCODE_API_KEY"),
            default_headers={
                "x-opencode-session": self.session_id,
                "User-Agent": LLM_USER_AGENT,
            },
        )

    def rotate_session(self) -> None:
        """Start a new conversation by rotating the session ID and rebuilding the client."""
        self.session_id = str(uuid.uuid4())
        self.client = self._create_openai_client()
        logger.info("LLM session rotated")

    def complete(self, messages: list, temperature: float = LLM_TEMPERATURE) -> str:
        """
        Run a chat completion and return the assistant message content.

        Args:
            messages: OpenAI-format message list ([{"role": ..., "content": ...}, ...])
            temperature: Sampling temperature

        Returns:
            The assistant's reply text

        Raises:
            Exception: Propagates API errors to the caller
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
