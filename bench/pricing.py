"""USD per 1M tokens: (input, cached_input, output). Checked Sept 2026; verify on
https://developers.openai.com/api/docs/pricing before quoting cost in the paper."""
PRICES = {
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
}


def cost_usd(model: str, tok_in: int, tok_cached: int, tok_out: int):
    p = PRICES.get(model)
    if p is None:
        return None
    uncached = max(0, tok_in - tok_cached)
    return (uncached * p[0] + tok_cached * p[1] + tok_out * p[2]) / 1e6
