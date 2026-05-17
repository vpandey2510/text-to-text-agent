import os
from openai import OpenAI


HF_API_KEY = os.getenv("HF_API_KEY")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY,
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Pro:novita",
    messages=[{
        "role": "user",
        "content": "Steps to learn python from scratch"
    }],
)

print(response.choices[0].message.content)
