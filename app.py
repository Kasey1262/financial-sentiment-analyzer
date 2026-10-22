import streamlit as st
from analyzer import analyze_sentiment

st.set_page_config(page_title="Financial News Sentiment Analyzer", page_icon="📈")

st.title("📈 Financial News Sentiment Analyzer")
st.caption("Powered by Claude API | Built by Kasey Yin")

news_text = st.text_area(
    "Paste a financial news article or headline:",
    height=200,
    placeholder="e.g. The Federal Reserve announced today it will hold interest rates steady...",
    key="news_input",
)

if st.button("Analyze Sentiment", type="primary"):
    st.session_state["analysis_error"] = None
    if not news_text.strip():
        st.warning("Please paste some news text first.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_sentiment(news_text)

        if result["sentiment"] == "error":
            st.session_state["analysis_error"] = result["summary"]
        else:
            st.session_state["last_analysis"] = {
                "news_text": news_text,
                "result": result,
            }

analysis_error = st.session_state.get("analysis_error")
if analysis_error:
    st.error(analysis_error)

last_analysis = st.session_state.get("last_analysis")
if last_analysis:
    st.subheader("Last successful analysis")
    if analysis_error:
        st.caption(
            "The latest request failed. Your last successful analysis is still shown below."
        )
    elif news_text != last_analysis["news_text"]:
        st.caption(
            "You are viewing the last analyzed text. Submit your edits to update the results."
        )

    with st.expander("Analyzed news", expanded=True):
        st.text(last_analysis["news_text"])

    result = last_analysis["result"]
    sentiment = result["sentiment"]
    if sentiment == "bullish":
        st.success(f"**Sentiment: {sentiment.upper()}** 🐂")
    elif sentiment == "bearish":
        st.error(f"**Sentiment: {sentiment.upper()}** 🐻")
    else:
        st.info(f"**Sentiment: {sentiment.upper()}** ➖")

    st.metric("Model confidence (self-assessed)", f"{result['confidence']}%")
    st.caption("This is the model's self-assessment, not a verified probability.")
    st.progress(result["confidence"] / 100)

    st.subheader("Key Factors")
    for index, factor in enumerate(result["key_factors"], start=1):
        st.write(f"{index}. {factor['factor']}")
        st.caption("Source quote")
        st.text(factor["quote"])

    st.subheader("Investment Interpretation")
    st.write(result["summary"])
