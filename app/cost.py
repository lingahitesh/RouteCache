def compute_cost(tokens_in: int, tokens_out: int, price_in: float, price_out: float) -> float:
    return tokens_in * price_in + tokens_out * price_out