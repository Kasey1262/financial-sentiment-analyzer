import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# 系统提示词：定义AI的角色和输出格式
# 这叫"结构化prompt设计" —— 强制AI返回固定格式的JSON，方便程序处理
SYSTEM_PROMPT = """You are a financial sentiment analysis assistant.
Analyze the financial news text provided by the user.

Respond ONLY with valid JSON in exactly this format, no other text:
{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": <number between 0 and 100>,
  "key_factors": [<list of 2-4 short strings describing key risk/driver factors>],
  "summary": "<one sentence investment-relevant interpretation>"
}"""


def analyze_sentiment(news_text: str) -> dict:
    """
    分析一段金融新闻的情绪。
    输入：新闻文本字符串
    输出：包含sentiment/confidence/key_factors/summary的字典
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": news_text}
        ]
    )

    raw_text = response.content[0].text

    # 剥掉可能存在的markdown代码块标记
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError:
        return {
            "sentiment": "error",
            "confidence": 0,
            "key_factors": [],
            "summary": "Failed to parse model response."
        }


# 测试代码：直接运行这个文件时会执行
if __name__ == "__main__":
    test_news = """
    The Federal Reserve announced today it will hold interest rates steady,
    citing persistent inflation concerns. Tech stocks rallied on the news,
    with the NASDAQ gaining 2.3% in afternoon trading.
    """

    result = analyze_sentiment(test_news)
    print(json.dumps(result, indent=2))