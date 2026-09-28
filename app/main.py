from fastapi import FastAPI
from pydantic import BaseModel
from config.config import load_config
from app.adapters.adapters import ProviderAdapter
from app.cost import compute_cost

app = FastAPI(title="RouteCache — Part 1 (pass-through baseline)")

config = load_config()
large_adapter = ProviderAdapter(config.models["large"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float = 0.0


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    response = large_adapter.call(messages, temperature=request.temperature)

    cost = compute_cost(
        response.tokens_in, response.tokens_out,
        config.models["large"].price_in, config.models["large"].price_out,
    )

    return {
        "choices": [
            {"message": {"role": "assistant", "content": response.text}}
        ],
        "x_routecache": {
            "cache": "disabled",
            "route": "large-model",
            "cost_usd": cost,
            "latency_ms": {"total": response.latency_ms},
        },
    }