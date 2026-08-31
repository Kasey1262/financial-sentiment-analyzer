# 📈 Financial News Sentiment Analyzer

A web app that analyzes financial news sentiment using the Anthropic Claude API.
Paste any financial news article or headline and get structured sentiment analysis
in seconds.

**[Live Demo](https://financial-sentiment-analyzer-eagekrw4sepf54rx35bgve.streamlit.app/)

## Features

- **Sentiment classification** — bullish / bearish / neutral with confidence score
- **Key factor extraction** — identifies 2-4 key market drivers or risks from the text
- **Investment interpretation** — one-sentence actionable summary
- **Structured JSON output** — enforced via prompt engineering for reliable parsing

## Tech Stack

- **Python** — core logic
- **Anthropic Claude API** — LLM-powered analysis with structured prompt design
- **Streamlit** — web interface and cloud deployment
- **python-dotenv** — local secrets management

## How It Works

1. User pastes financial news text
2. A structured system prompt instructs Claude to return strict JSON
   (sentiment, confidence, key factors, summary)
3. Response is cleaned (markdown fence stripping) and parsed with error handling
4. Results are rendered with color-coded sentiment indicators

## Validation

To check that the classifier actually works — not just that it runs — I built a
small **hand-labeled evaluation set** ([`labeled_samples.json`](labeled_samples.json))
of 30 financial-news items across the three classes and wrote
[`evaluate.py`](evaluate.py), which runs each item through `analyze_sentiment()`
and reports accuracy, a confusion matrix, per-class precision/recall, and a
confidence-vs-correctness check.

**Reproduce:** `python3 evaluate.py` (from the repo root, with your `.env` set).

### Results (30 hand-labeled items)

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

- **Perfect precision on bullish and bearish (1.00):** the model never confused a
  positive story for a negative one, or vice versa. Both of its errors were
  directional items it labeled *neutral* — so its failure mode is being
  **conservative**, not reckless, which is the safer direction to err for a
  sentiment signal.
- **Both misses were deliberately mixed-signal cases** (e.g. earnings and price
  action pointing in opposite directions), and the model reported low confidence
  on both — it was uncertain exactly where a human would be.
- **Confidence carries real information:** average confidence was **83.8 when
  correct** vs **72.0 when wrong**, so higher confidence does track with being
  right on this set.

### Limitations & next steps

This is a **small, single-annotator** evaluation, so the numbers are indicative,
not a rigorous benchmark. Planned next steps: (1) scale to a larger labelled set
for statistically meaningful per-class metrics; (2) add a second annotator and
measure agreement on the ambiguous cases; (3) formally test confidence
calibration; and (4) move schema enforcement to the API's structured-output /
tool-use feature to further reduce parse risk.

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
