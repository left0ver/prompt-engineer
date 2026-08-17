"""Unit tests for the shared model wrapper; no network calls are made."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from code.common import call_model


class CallModelTests(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_MODEL": "test-model"})
    @patch("code.common._get_client")
    def test_call_model_builds_request_and_returns_text(self, get_client: MagicMock) -> None:
        create = get_client.return_value.chat.completions.create
        create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  PRODUCT  "))]
        )

        result = call_model(
            "测试提示",
            system_prompt="系统提示",
        )

        self.assertEqual(result, "PRODUCT")
        create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "测试提示"},
            ],
        )

    def test_call_model_rejects_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            call_model("   ")

    @patch.dict(os.environ, {"LLM_MODEL": "test-model"})
    @patch("code.common._get_client")
    def test_call_model_passes_optional_temperature(self, get_client: MagicMock) -> None:
        create = get_client.return_value.chat.completions.create
        create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        call_model("测试提示", temperature=0.7)

        self.assertEqual(create.call_args.kwargs["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
