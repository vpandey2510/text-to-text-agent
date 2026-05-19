import os

from dotenv import load_dotenv
from openai import OpenAI

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro:novita"
DEFAULT_PROMPT = "Steps to learn python from scratch"


def get_api_key() -> str | None:
    load_dotenv()
    return os.getenv("HF_API_KEY")


def create_client(api_key: str, base_url: str = HF_ROUTER_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def chat(client: OpenAI, prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main(prompt: str = DEFAULT_PROMPT, model: str = DEFAULT_MODEL) -> None:
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "HF_API_KEY is not set. Add it to .env or export it in your shell."
        )
    client = create_client(api_key)
    print(chat(client, prompt, model=model))


if __name__ == "__main__":
    main()
