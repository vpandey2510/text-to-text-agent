from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI


def _mock_chat_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestGetApiKey:
    @patch("text_agent.load_dotenv")
    @patch("text_agent.os.getenv", return_value="hf_test_token")
    def test_returns_key_from_environment(self, mock_getenv, mock_load_dotenv, text_agent):
        assert text_agent.get_api_key() == "hf_test_token"
        mock_load_dotenv.assert_called_once()
        mock_getenv.assert_called_once_with("HF_API_KEY")

    @patch("text_agent.load_dotenv")
    @patch("text_agent.os.getenv", return_value=None)
    def test_returns_none_when_missing(self, mock_getenv, mock_load_dotenv, text_agent):
        assert text_agent.get_api_key() is None


class TestCreateClient:
    def test_uses_huggingface_router_base_url(self, text_agent):
        with patch.object(text_agent, "OpenAI", return_value=MagicMock(spec=OpenAI)) as mock_openai:
            client = text_agent.create_client("hf_test_token")

        mock_openai.assert_called_once_with(
            base_url=text_agent.HF_ROUTER_BASE_URL,
            api_key="hf_test_token",
        )
        assert client is mock_openai.return_value

    def test_accepts_custom_base_url(self, text_agent):
        with patch.object(text_agent, "OpenAI", return_value=MagicMock(spec=OpenAI)) as mock_openai:
            text_agent.create_client("hf_test_token", base_url="https://custom.example/v1")

        mock_openai.assert_called_once_with(
            base_url="https://custom.example/v1",
            api_key="hf_test_token",
        )


class TestChat:
    def test_sends_prompt_and_returns_content(self, text_agent):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response(
            "Start with variables and loops."
        )

        result = text_agent.chat(client, "Steps to learn python from scratch")

        client.chat.completions.create.assert_called_once_with(
            model=text_agent.DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Steps to learn python from scratch"}],
        )
        assert result == "Start with variables and loops."

    def test_uses_custom_model(self, text_agent):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("ok")

        text_agent.chat(client, "Hi", model="other-model")

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "other-model"


class TestMain:
    @patch("text_agent.print")
    @patch("text_agent.chat", return_value="Model reply")
    @patch("text_agent.create_client")
    @patch("text_agent.get_api_key", return_value="hf_test_token")
    def test_prints_chat_response(
        self, mock_get_api_key, mock_create_client, mock_chat, mock_print, text_agent
    ):
        text_agent.main(prompt="Hello", model="test-model")

        mock_get_api_key.assert_called_once()
        mock_create_client.assert_called_once_with("hf_test_token")
        mock_chat.assert_called_once_with(
            mock_create_client.return_value,
            "Hello",
            model="test-model",
        )
        mock_print.assert_called_once_with("Model reply")

    @patch("text_agent.get_api_key", return_value=None)
    def test_raises_when_api_key_missing(self, mock_get_api_key, text_agent):
        with pytest.raises(ValueError, match="HF_API_KEY is not set"):
            text_agent.main()
