import os
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, provider="ollama", model=None, api_key=None):
        self.provider = provider
        if provider == "ollama":
            self.model = model or "qwen2.5:7b"
            self.url = "http://localhost:11434/api/chat"
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key 未提供")
        else:
            raise ValueError("provider 必須為 'ollama' 或 'openai'")

    def chat(self, messages, temperature=0.3):
        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            }
            resp = requests.post(self.url, json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        else:
            import openai
            openai.api_key = self.api_key
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return resp.choices[0].message.content