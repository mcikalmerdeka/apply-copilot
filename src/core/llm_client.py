"""
Shared LLM client for ApplyCopilot.

Centralizes OpenAI client construction and the chat-completion call pattern
used by the cover letter generator and the employer Q&A chatbot.

Text generation runs on GPT-6 Luna with low reasoning effort. Reasoning models
do not accept sampling controls such as ``temperature`` when reasoning is
enabled, so the request intentionally omits them.
See https://developers.openai.com/api/docs/models/gpt-6-luna
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.config.logging_config import setup_logger
from src.config.settings import LLM_MODEL, LLM_REASONING_EFFORT

load_dotenv()
logger = setup_logger(__name__)


class LLMClient:
    """OpenAI chat client for GPT-6 Luna with configurable reasoning effort."""

    def __init__(self, llm_model: str = LLM_MODEL,
                 reasoning_effort: str = LLM_REASONING_EFFORT):
        """
        Initialize the LLM client.

        Args:
            llm_model: OpenAI model name
            reasoning_effort: Reasoning effort passed to the model
        """
        self.model = llm_model
        self.reasoning_effort = reasoning_effort
        self.client = self._create_openai_client()
        logger.info(
            f"Initialized LLMClient with model: {llm_model} "
            f"(reasoning effort: {reasoning_effort})"
        )

    def _create_openai_client(self) -> OpenAI:
        """Create an OpenAI client, failing early when the API key is missing."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )
        return OpenAI(api_key=api_key)

    def complete(self, messages: list) -> str:
        """
        Run a chat completion and return the assistant message content.

        Args:
            messages: OpenAI-format message list ([{"role": ..., "content": ...}, ...])

        Returns:
            The assistant's reply text

        Raises:
            Exception: Propagates API errors to the caller
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            reasoning_effort=self.reasoning_effort,
        )
        return response.choices[0].message.content
