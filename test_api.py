import os
from dotenv import load_dotenv
from anthropic import Anthropic

# 从.env文件读取API key（这样key不会写死在代码里）
load_dotenv()

# 创建API客户端
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# 发送第一个请求
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "Explain financial market sentiment analysis in one sentence."}
    ]
)

# 打印Claude的回复
print(response.content[0].text)