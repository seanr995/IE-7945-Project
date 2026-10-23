"""Matched-granularity grid around the agentic topic count (a reference granularity, NOT ground truth)."""
from __future__ import annotations


def matched_grid(n_agentic: int, n_statements: int | None = None, rel=0.20, min_delta=2) -> dict:
    """N_LOW / N_AGENTIC / N_HIGH.

    delta = max(min_delta, round(rel * N)): +-20% normally; for small N a +-2 neighbourhood so the grid never collapses
    into duplicates or meaningless counts. Counts are integers >= 2, unique, and (if given) <= n_statements // 5.
    """
    n = int(n_agentic)
    if n < 2:
        raise ValueError(f"N_AGENTIC={n} is too small to define a matched-granularity grid")
    delta = max(min_delta, int(round(rel * n)))
    low, high = max(2, n - delta), n + delta
    if n_statements:
        high = min(high, max(n + 1, n_statements // 5))
    grid = sorted({low, n, high})
    rule = (f"delta = max({min_delta}, round({rel} * N)) = {delta}" +
            (" (small-N neighbourhood)" if delta == min_delta and round(rel * n) < min_delta else " (+-20%)"))
    return {"N_LOW": grid[0], "N_AGENTIC": n, "N_HIGH": grid[-1], "grid": grid, "rule": rule}
