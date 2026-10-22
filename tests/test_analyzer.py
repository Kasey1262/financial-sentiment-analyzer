"""Offline contract tests: all Anthropic clients are mocked."""

import copy
import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

import analyzer


NEWS = "Revenue rose 12%.\nManagement raised its outlook."
VALID_RESULT = {
    "sentiment": "bullish",
    "confidence": 87,
    "key_factors": [
        {"factor": "Revenue growth", "quote": "Revenue rose 12%."},
        {"factor": "Improved outlook", "quote": "Management raised its outlook."},
    ],
    "summary": "The text expresses optimism about growth and the outlook.",
}


def response_for(payload, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.client_patch = patch("analyzer.Anthropic")
        self.client_factory = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.client = self.client_factory.return_value.__enter__.return_value
        self.client.messages.create.return_value = response_for(VALID_RESULT)

    def assert_error(self, result):
        self.assertEqual(
            set(result), {"sentiment", "confidence", "key_factors", "summary"}
        )
        self.assertEqual(result["sentiment"], "error")
        self.assertEqual(result["confidence"], 0)
        self.assertEqual(result["key_factors"], [])
        self.assertTrue(result["summary"])
        self.assertNotIn("secret-provider-payload", result["summary"])
        self.assertNotIn("test-key", result["summary"])

    def test_import_does_not_require_key_or_create_client(self):
        # Patch at the SDK boundary before import in a clean interpreter. This
        # also prevents an accidental real client/network call in this test.
        script = """import os
import anthropic
import dotenv
os.environ.pop("ANTHROPIC_API_KEY", None)
dotenv.load_dotenv = lambda: None
def fail_if_created(*args, **kwargs):
    raise AssertionError("A client was created during import")
anthropic.Anthropic = fail_if_created
import analyzer
assert analyzer.analyze_sentiment("Some financial news.")["sentiment"] == "error"
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_success_returns_evidence_and_uses_structured_output(self):
        result = analyzer.analyze_sentiment(NEWS)
        self.assertEqual(result, VALID_RESULT)
        call = self.client.messages.create.call_args.kwargs
        self.assertEqual(call["model"], "claude-sonnet-4-5")
        self.assertEqual(call["messages"], [{"role": "user", "content": NEWS}])
        output_format = call["output_config"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertFalse(output_format["schema"]["additionalProperties"])
        self.assertEqual(
            set(output_format["schema"]["required"]),
            {"sentiment", "confidence", "key_factors", "summary"},
        )
        self.client_factory.assert_called_once_with(
            api_key="test-key", timeout=30.0, max_retries=1
        )
        self.client_factory.return_value.__exit__.assert_called_once()

    def test_numeric_confidence_boundaries_are_accepted(self):
        for confidence in (0, 100, 63.5):
            with self.subTest(confidence=confidence):
                payload = copy.deepcopy(VALID_RESULT)
                payload["confidence"] = confidence
                self.client.messages.create.return_value = response_for(payload)
                result = analyzer.analyze_sentiment(NEWS)
                self.assertEqual(result["confidence"], confidence)

    def test_invalid_confidence_is_not_coerced(self):
        for confidence in (True, False, "80", None, -1, 101, float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                payload = copy.deepcopy(VALID_RESULT)
                payload["confidence"] = confidence
                self.client.messages.create.return_value = response_for(payload)
                self.assert_error(analyzer.analyze_sentiment(NEWS))

    def test_invalid_labels_and_extra_fields_are_rejected(self):
        for changes in (
            {"sentiment": "positive"},
            {"sentiment": "BULLISH"},
            {"sentiment": None},
            {"unrequested": "secret-provider-payload"},
        ):
            with self.subTest(changes=changes):
                payload = dict(VALID_RESULT, **changes)
                self.client.messages.create.return_value = response_for(payload)
                self.assert_error(analyzer.analyze_sentiment(NEWS))

    def test_missing_fields_nonobjects_and_invalid_json_are_rejected(self):
        missing = copy.deepcopy(VALID_RESULT)
        del missing["summary"]
        responses = [
            response_for(missing),
            response_for(None),
            response_for([]),
            SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="secret-provider-payload")],
            ),
        ]
        for response in responses:
            with self.subTest(response=response):
                self.client.messages.create.return_value = response
                self.assert_error(analyzer.analyze_sentiment(NEWS))

    def test_factor_count_and_old_string_factors_are_rejected(self):
        for factors in (
            [],
            VALID_RESULT["key_factors"][:1],
            VALID_RESULT["key_factors"] * 3,
            ["Revenue grew", "Outlook improved"],
        ):
            with self.subTest(factors=factors):
                payload = dict(VALID_RESULT, key_factors=factors)
                self.client.messages.create.return_value = response_for(payload)
                self.assert_error(analyzer.analyze_sentiment(NEWS))

    def test_blank_or_wrong_type_text_fields_are_rejected(self):
        for field, value in (
            ("summary", ""),
            ("summary", " \n"),
            ("summary", 42),
            ("factor", ""),
            ("factor", "  "),
            ("factor", 42),
            ("quote", ""),
            ("quote", " \n"),
            ("quote", 42),
        ):
            with self.subTest(field=field, value=value):
                payload = copy.deepcopy(VALID_RESULT)
                if field == "summary":
                    payload[field] = value
                else:
                    payload["key_factors"][0][field] = value
                self.client.messages.create.return_value = response_for(payload)
                self.assert_error(analyzer.analyze_sentiment(NEWS))

    def test_quotes_must_be_exact_source_substrings(self):
        for quote in (
            "Revenue rose 50%.",
            "revenue rose 12%.",
            "Revenue rose 12%. Management raised its outlook.",
            "Revenue rose 12%...",
        ):
            with self.subTest(quote=quote):
                payload = copy.deepcopy(VALID_RESULT)
                payload["key_factors"][0]["quote"] = quote
                self.client.messages.create.return_value = response_for(payload)
                result = analyzer.analyze_sentiment(NEWS)
                self.assert_error(result)
                self.assertIn("could not be found", result["summary"])

    def test_quotes_preserve_unicode_punctuation_and_whitespace(self):
        news = "利润上涨 12%。\n  Outlook improved—management said so."
        payload = copy.deepcopy(VALID_RESULT)
        payload["key_factors"] = [
            {"factor": "Profit growth", "quote": "利润上涨 12%。"},
            {"factor": "Improved outlook", "quote": "\n  Outlook improved—management said so."},
        ]
        self.client.messages.create.return_value = response_for(payload)
        self.assertEqual(analyzer.analyze_sentiment(news), payload)

    def test_invalid_input_does_not_call_provider(self):
        for text in ("", " \n", None, 42):
            with self.subTest(text=text):
                self.assert_error(analyzer.analyze_sentiment(text))
        self.client_factory.assert_not_called()

    def test_missing_or_blank_key_does_not_create_client(self):
        for value in (None, "", "  "):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop("ANTHROPIC_API_KEY", None)
                else:
                    os.environ["ANTHROPIC_API_KEY"] = value
                result = analyzer.analyze_sentiment(NEWS)
                self.assert_error(result)
                self.assertIn("not configured", result["summary"])
        self.client_factory.assert_not_called()

    def test_expected_provider_failures_are_friendly_and_hide_payloads(self):
        request = Mock()
        cases = [
            (
                AuthenticationError(
                    "secret-provider-payload",
                    response=Mock(status_code=401, headers={}, request=request),
                    body={"secret": "secret-provider-payload"},
                ),
                "authenticate",
            ),
            (
                RateLimitError(
                    "secret-provider-payload",
                    response=Mock(status_code=429, headers={}, request=request),
                    body=None,
                ),
                "busy",
            ),
            (APITimeoutError(request=request), "too long"),
            (
                APIConnectionError(message="secret-provider-payload", request=request),
                "connect",
            ),
            (
                APIStatusError(
                    "secret-provider-payload",
                    response=Mock(status_code=503, headers={}, request=request),
                    body=None,
                ),
                "unavailable",
            ),
            (
                APIError("secret-provider-payload", request=request, body=None),
                "unavailable",
            ),
        ]
        for error, message in cases:
            with self.subTest(error=type(error).__name__):
                self.client.messages.create.side_effect = error
                result = analyzer.analyze_sentiment(NEWS)
                self.assert_error(result)
                self.assertIn(message, result["summary"])

    def test_client_creation_failures_use_the_same_error_contract(self):
        self.client_factory.side_effect = APITimeoutError(request=Mock())
        result = analyzer.analyze_sentiment(NEWS)
        self.assert_error(result)
        self.assertIn("too long", result["summary"])

    def test_refusal_and_truncation_are_checked_before_json_parsing(self):
        for stop_reason, message in (
            ("refusal", "could not complete"),
            ("max_tokens", "cut short"),
            ("tool_use", "did not finish"),
            (None, "did not finish"),
        ):
            with self.subTest(stop_reason=stop_reason):
                self.client.messages.create.return_value = SimpleNamespace(
                    stop_reason=stop_reason,
                    content=[SimpleNamespace(type="text", text="secret-provider-payload")],
                )
                result = analyzer.analyze_sentiment(NEWS)
                self.assert_error(result)
                self.assertIn(message, result["summary"])

    def test_missing_or_multiple_text_blocks_are_rejected(self):
        for content in (
            [],
            [SimpleNamespace(type="tool_use")],
            [
                SimpleNamespace(type="text", text=json.dumps(VALID_RESULT)),
                SimpleNamespace(type="text", text=json.dumps(VALID_RESULT)),
            ],
        ):
            with self.subTest(content=content):
                self.client.messages.create.return_value = SimpleNamespace(
                    stop_reason="end_turn", content=content
                )
                self.assert_error(analyzer.analyze_sentiment(NEWS))


if __name__ == "__main__":
    unittest.main()
