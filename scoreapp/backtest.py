"""Walk-forward backtesting: is the model actually any good?

For every match in the target season, the model is fitted using only matches
played *before* that date (refitted at most weekly), so predictions are made
exactly as they would have been in real time — no lookahead.

Metrics
-------
RPS (ranked probability score) is the standard metric for 1X2 football
predictions: it respects outcome ordering (home/draw/away), lower is better.
Brier score and log-loss are reported too. Bookmaker (Bet365) odds from the
same CSVs are converted to probabilities and scored identically — beating or
getting close to the bookmaker is the realistic bar for a good model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DixonColesModel

OUTCOMES = ["H", "D", "A"]


def rps(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    """Ranked probability score per match. probs: (n, 3) ordered H, D, A."""
    actual = np.zeros_like(probs)
    actual[np.arange(len(probs)), outcome_idx] = 1.0
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(actual, axis=1)
    return (cum_diff[:, :-1] ** 2).sum(axis=1) / (probs.shape[1] - 1)


def brier(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    actual = np.zeros_like(probs)
    actual[np.arange(len(probs)), outcome_idx] = 1.0
    return ((probs - actual) ** 2).sum(axis=1)


def log_loss(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    p = probs[np.arange(len(probs)), outcome_idx]
    return -np.log(np.clip(p, 1e-12, None))


def walk_forward(all_matches: pd.DataFrame, season: str, xi: float | None = None,
                 refit_days: int = 7) -> pd.DataFrame:
    """Predict every match of `season` with weekly refits on prior data only."""
    target = all_matches[all_matches["Season"] == season].sort_values("Date")
    model = DixonColesModel(**({"xi": xi} if xi is not None else {}))
    last_fit: pd.Timestamp | None = None
    rows = []
    for date, day in target.groupby("Date"):
        if last_fit is None or (date - last_fit).days >= refit_days:
            model.fit(all_matches, as_of=date)
            last_fit = date
        for _, m in day.iterrows():
            try:
                p = model.outcome_probs(m["HomeTeam"], m["AwayTeam"])
            except KeyError:
                continue  # newly promoted team with no history yet
            row = {
                "Date": date, "HomeTeam": m["HomeTeam"], "AwayTeam": m["AwayTeam"],
                "FTR": m["FTR"], "pH": p["H"], "pD": p["D"], "pA": p["A"],
            }
            if {"B365H", "B365D", "B365A"}.issubset(m.index) and pd.notna(m["B365H"]):
                inv = np.array([1 / m["B365H"], 1 / m["B365D"], 1 / m["B365A"]])
                row["bH"], row["bD"], row["bA"] = inv / inv.sum()  # strip the overround
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(preds: pd.DataFrame) -> pd.DataFrame:
    outcome_idx = preds["FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    result = {}
    model_p = preds[["pH", "pD", "pA"]].to_numpy()
    result["model"] = {
        "matches": len(preds),
        "rps": rps(model_p, outcome_idx).mean(),
        "brier": brier(model_p, outcome_idx).mean(),
        "log_loss": log_loss(model_p, outcome_idx).mean(),
        "accuracy": (model_p.argmax(axis=1) == outcome_idx).mean(),
    }
    if "bH" in preds.columns:
        book = preds.dropna(subset=["bH"])
        book_idx = book["FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
        book_p = book[["bH", "bD", "bA"]].to_numpy()
        result["bookmaker (B365)"] = {
            "matches": len(book),
            "rps": rps(book_p, book_idx).mean(),
            "brier": brier(book_p, book_idx).mean(),
            "log_loss": log_loss(book_p, book_idx).mean(),
            "accuracy": (book_p.argmax(axis=1) == book_idx).mean(),
        }
    return pd.DataFrame(result).T.round(4)
