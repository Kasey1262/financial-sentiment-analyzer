import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
NEWS = "Revenue rose 12% this quarter. Operating margins improved to 18%."
SUCCESS = {
    "sentiment": "bullish",
    "confidence": 82,
    "key_factors": [
        {"factor": "Higher revenue", "quote": "Revenue rose 12%"},
        {"factor": "Improved margins", "quote": "Operating margins improved to 18%"},
    ],
    "summary": "Revenue and margin growth support a bullish interpretation.",
}
ERROR = {
    "sentiment": "error",
    "confidence": 0,
    "key_factors": [],
    "summary": "The analysis service is temporarily unavailable. Please try again.",
}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("analyzer.analyze_sentiment", return_value=deepcopy(SUCCESS))
        self.analyze = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.app = AppTest.from_file(str(APP_PATH)).run()

    def submit(self, news_text):
        self.app.text_area[0].input(news_text).run()
        self.app.button[0].click().run()
        self.assertEqual(len(self.app.exception), 0)

    def text_values(self):
        return [element.value for element in self.app.text]

    def test_analysis_displays_original_news_and_exact_source_quotes(self):
        self.submit(NEWS)

        self.analyze.assert_called_once_with(NEWS)
        self.assertIn(NEWS, self.text_values())
        for factor in SUCCESS["key_factors"]:
            self.assertIn(factor["quote"], self.text_values())
            self.assertTrue(
                any(factor["factor"] in element.value for element in self.app.markdown)
            )
        self.assertEqual(self.app.metric[0].label, "Model confidence (self-assessed)")
        self.assertEqual(self.app.metric[0].value, "82%")
        self.assertTrue(
            any("not a verified probability" in item.value for item in self.app.caption)
        )

    def test_last_result_survives_input_edit_and_another_rerun(self):
        self.submit(NEWS)
        edited_news = "Shares fell after the company lowered its outlook."

        self.app.text_area[0].input(edited_news).run()
        self.app.run()

        self.assertEqual(len(self.app.exception), 0)
        self.analyze.assert_called_once_with(NEWS)
        self.assertEqual(self.app.text_area[0].value, edited_news)
        self.assertIn(NEWS, self.text_values())
        self.assertNotIn(edited_news, self.text_values())
        self.assertEqual(self.app.metric[0].value, "82%")
        self.assertTrue(
            any("Submit your edits" in item.value for item in self.app.caption)
        )

    def test_failed_request_preserves_previous_success_with_clear_label(self):
        self.submit(NEWS)
        self.analyze.return_value = ERROR

        self.submit("The company reported another quarter of results.")
        self.app.run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.analyze.call_count, 2)
        self.assertIn(ERROR["summary"], [item.value for item in self.app.error])
        self.assertIn(NEWS, self.text_values())
        self.assertEqual(self.app.metric[0].value, "82%")
        self.assertTrue(
            any("latest request failed" in item.value for item in self.app.caption)
        )

    def test_first_failure_shows_error_without_analysis(self):
        self.analyze.return_value = ERROR

        self.submit(NEWS)

        self.assertIn(ERROR["summary"], [item.value for item in self.app.error])
        self.assertEqual(len(self.app.metric), 0)
        self.assertEqual(len(self.app.text), 0)

    def test_empty_submission_warns_without_calling_analyzer(self):
        self.submit(" \n ")

        self.analyze.assert_not_called()
        self.assertEqual(self.app.warning[0].value, "Please paste some news text first.")
        self.assertEqual(len(self.app.metric), 0)
        self.assertEqual(len(self.app.text), 0)

    def test_successful_resubmission_replaces_source_and_result(self):
        self.submit(NEWS)
        next_news = "Profit fell 5%. Management cut its outlook."
        self.analyze.return_value = {
            "sentiment": "bearish",
            "confidence": 75,
            "key_factors": [
                {"factor": "Lower profit", "quote": "Profit fell 5%"},
                {"factor": "Weaker outlook", "quote": "Management cut its outlook"},
            ],
            "summary": "Lower profit and guidance support a bearish interpretation.",
        }

        self.submit(next_news)

        self.assertEqual(self.analyze.call_count, 2)
        self.assertNotIn(NEWS, self.text_values())
        self.assertIn(next_news, self.text_values())
        self.assertIn("Profit fell 5%", self.text_values())
        self.assertEqual(self.app.metric[0].value, "75%")


if __name__ == "__main__":
    unittest.main()
