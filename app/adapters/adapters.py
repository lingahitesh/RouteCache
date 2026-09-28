import os
import time
from dataclasses import dataclass
import httpx

@dataclass
class ProviderResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float

class ProviderAdapter:
    def __init__(self, model_config):
        self.model_config = model_config
        self.client = httpx.Client(base_url=model_config.base_url, timeout=30.0)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.model_config.api_key_env:
            api_key = os.environ.get(self.model_config.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing environment variable: {self.model_config.api_key_env}. "
                    f"Check your .env file."
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def call(self, messages: list[dict], temperature: float = 0.0) -> ProviderResponse:
        start = time.time()
        payload = {
            "model": self.model_config.name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        response = self.client.post("/chat/completions", json=payload, headers=self._headers())
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.time() - start) * 1000

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        return ProviderResponse(
            text=text, tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms
        )