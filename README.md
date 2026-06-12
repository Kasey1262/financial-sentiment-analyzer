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
