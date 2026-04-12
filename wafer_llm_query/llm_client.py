import requests

class LLMClient:
    def __init__(self, provider="ollama", model=None, api_key=None):
        """
        Only supports the Ollama backend.
        The provider parameter is reserved for compatibility purposes only; passing a value other than 'ollama' will throw an error.
        """
        if provider != "ollama":
            raise ValueError("Only 'ollama' provider is supported. Please set provider='ollama'.")
        self.model = model or "llama3.2:3b"
        self.url = "http://localhost:11434/api/chat"

    def chat(self, messages, temperature=0.3):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        resp = requests.post(self.url, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]