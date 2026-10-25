"""Behavior tests for the shared LLM client."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.core.llm_client import LLMClient


class LLMClientTest(unittest.TestCase):
    @staticmethod
    def _completion(content: str) -> MagicMock:
        response = MagicMock()
        response.choices[0].message.content = content
        return response

    @patch("src.core.llm_client.OpenAI")
    def test_complete_uses_gpt_6_luna_with_low_reasoning(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value = (
            self._completion("Hello")
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            result = client.complete([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "Hello")
        mock_openai.assert_called_once_with(api_key="test-key")

        request = mock_openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-6-luna")
        self.assertEqual(request["reasoning_effort"], "low")
        # Reasoning models reject sampling controls when reasoning is enabled.
        self.assertNotIn("temperature", request)

    @patch("src.core.llm_client.OpenAI")
    def test_missing_api_key_fails_fast(self, mock_openai):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                LLMClient()

        mock_openai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
