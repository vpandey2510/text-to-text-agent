# Text-to-Text Agent

A minimal Python script that sends a prompt to a Hugging Face–hosted model via the OpenAI-compatible API and prints the reply.

## Requirements

- Python 3.10+
- A [Hugging Face](https://huggingface.co/) account with an API token

## Setup

1. Clone or copy this folder.

2. Install dependencies:

   ```bash
   pip install openai python-dotenv
   ```

3. Create your local environment file from the example:

   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and set your Hugging Face token:

   ```
   HF_API_KEY=hf_your_token_here
   ```

5. Save `.env` in this folder. The script loads it automatically via `python-dotenv`.

   Use this format (no spaces around `=`):

   ```
   HF_API_KEY=hf_your_token_here
   ```

   `.env` is listed in `.gitignore` and will not be committed.

## Usage

```bash
python3 text-agent.py
```

The script calls `deepseek-ai/DeepSeek-V4-Pro:novita` through the Hugging Face router and prints the model response.

To change the prompt or model, edit `text-agent.py`.

## Project layout

```
.
├── .env.example    # Template for secrets (safe to commit)
├── .gitignore      # Ignores .env and Python cache files
├── README.md
└── text-agent.py   # Main script
```

## Security

Never commit `.env` or paste API keys into git. Use `.env.example` as a template only.
