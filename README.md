# 📈 Financial News Sentiment Analyzer

A web app that analyzes financial news sentiment using the Anthropic Claude API.
Paste any financial news article or headline and get structured sentiment analysis
in seconds.

**[Live Demo](https://financial-sentiment-analyzer-eagekrw4sepf54rx35bgve.streamlit.app/)**

## Features

- **Sentiment classification** — bullish / bearish / neutral with a model self-assessed confidence score
- **Traceable factors** — 2-4 drivers or risks, each paired with an exact quote from the submitted news
- **Interpretation** — a short summary of the supplied news, including mixed signals when present
- **Validated output** — API-level JSON schema plus local checks of fields, ranges, and source quotes
- **Persistent results** — the last successful analysis stays visible while editing or after a failed request, with its original news clearly shown
- **Recoverable errors** — readable messages for missing credentials, service failures, invalid results, and unverifiable quotes

## Tech Stack

- **Python** — core logic
- **Anthropic Claude API** — LLM-powered analysis with structured prompt design
- **Streamlit** — web interface and cloud deployment
- **python-dotenv** — local secrets management
- **Pydantic** — strict response validation

## How It Works

1. User pastes financial news text
2. Claude receives a JSON schema for sentiment, confidence, factor/quote pairs, and summary
3. The response is checked for valid fields, labels, finite confidence in 0-100,
   2-4 factors, and nonblank text; every quote must occur verbatim in the input
4. Streamlit saves a successful result and its submitted news in session state,
   then renders the interpretation and supporting quotes independently of the button

Matching a quote verifies that it came from the input, not that the factor's
interpretation is correct. The confidence score is model self-assessment; it is
not a calibrated probability of correctness or future investment returns.

`analyze_sentiment(text)` still returns a dictionary with `sentiment`,
`confidence`, `key_factors`, and `summary`. Each successful `key_factors` item
is now `{"factor": "...", "quote": "exact text from the news"}` rather than a
string. Expected failures use `sentiment: "error"`, a readable `summary`, zero
confidence, and an empty factor list. The existing evaluation script uses the
unchanged sentiment/confidence fields.

## Offline checks

```bash
python3 -m unittest discover -s tests -v
```

These tests use mocked provider responses and Streamlit's AppTest. They require
no API key and make no model calls. They cover response validation, source-quote
matching, expected service failures, rendering, and retention of the previous
successful result. Session state lasts for the current browser connection; it
does not provide history across browser reloads or separate sessions.

## Validation

To check that the classifier actually works — not just that it runs — I built a
small **hand-labeled evaluation set** ([`labeled_samples.json`](labeled_samples.json))
of 30 financial-news items across the three classes and wrote
[`evaluate.py`](evaluate.py), which runs each item through `analyze_sentiment()`
and reports accuracy, a confusion matrix, per-class precision/recall, and a
confidence-vs-correctness check.

**Reproduce:** `python3 evaluate.py` (from the repo root, with your `.env` set).

### Historical baseline results (30 hand-labeled items)

The results below were reported for the earlier prompt and output format. They
have not been rerun for the evidence-backed schema in this revision. Running
`evaluate.py` uses the configured Claude account and incurs API usage; offline
tests do not establish model accuracy.

**Overall accuracy: 28/30 = 93.3%** — 0 parse errors (the JSON cleaning/error
handling held on every response).

Confusion matrix (rows = my label, columns = model prediction):

| true \ pred | bullish | bearish | neutral |
|-------------|:-------:|:-------:|:-------:|
| **bullish** |   10    |    0    |    1    |
| **bearish** |    0    |    9    |    1    |
| **neutral** |    0    |    0    |    9    |

Per-class:

| class   | precision | recall |  f1  | support |
|---------|:---------:|:------:|:----:|:-------:|
| bullish |   1.00    |  0.91  | 0.95 |   11    |
| bearish |   1.00    |  0.90  | 0.95 |   10    |
| neutral |   0.82    |  1.00  | 0.90 |    9    |

### What the results show

- **Precision on bullish and bearish was 1.00 on this set:** both errors were
  directional items labeled *neutral*. This small result does not establish a
  generally safer investment signal.
- **Both misses were deliberately mixed-signal cases** (e.g. earnings and price
  action pointing in opposite directions), and the model reported low confidence
  on both — it was uncertain exactly where a human would be.
- **Confidence differed between outcomes:** average confidence was **83.8 when
  correct** vs **72.0 when wrong**. Comparing these averages does not establish
  probability calibration.

### Limitations & next steps

This is a **small, single-annotator** evaluation, so the numbers are indicative,
not a rigorous benchmark. Planned next steps: (1) scale to a larger labelled set
for statistically meaningful per-class metrics; (2) add a second annotator and
measure agreement on the ambiguous cases; (3) formally test confidence
calibration; and (4) rerun the evaluation after prompt or schema changes. API
schema enforcement and local output validation are now implemented.

## Run Locally

```bash
git clone https://github.com/Kasey1262/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create a .env file with: ANTHROPIC_API_KEY=your_key_here
streamlit run app.py
```

## Author

**Kasey Yin** — Mathematics @ University of Waterloo
