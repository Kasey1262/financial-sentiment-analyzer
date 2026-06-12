import streamlit as st
from analyzer import analyze_sentiment

# 页面配置
st.set_page_config(page_title="Financial News Sentiment Analyzer", page_icon="📈")

st.title("📈 Financial News Sentiment Analyzer")
st.caption("Powered by Claude API | Built by Kasey Yin")

# 文本输入框
news_text = st.text_area(
    "Paste a financial news article or headline:",
    height=200,
    placeholder="e.g. The Federal Reserve announced today it will hold interest rates steady..."
)

# 分析按钮
if st.button("Analyze Sentiment", type="primary"):
    if not news_text.strip():
        st.warning("Please paste some news text first.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_sentiment(news_text)

        if result["sentiment"] == "error":
            st.error(result["summary"])
        else:
            # 根据情绪选颜色
            sentiment = result["sentiment"]
            if sentiment == "bullish":
                st.success(f"**Sentiment: {sentiment.upper()}** 🐂")
            elif sentiment == "bearish":
                st.error(f"**Sentiment: {sentiment.upper()}** 🐻")
            else:
                st.info(f"**Sentiment: {sentiment.upper()}** ➖")

            # 置信度进度条
            st.metric("Confidence", f"{result['confidence']}%")
            st.progress(result["confidence"] / 100)

            # 关键因素
            st.subheader("Key Factors")
            for factor in result["key_factors"]:
                st.markdown(f"- {factor}")

            # 总结
            st.subheader("Investment Interpretation")
            st.write(result["summary"])