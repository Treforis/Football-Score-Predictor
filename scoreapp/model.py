"""Dixon-Coles match prediction model.

Each team gets an attack and defence rating; goal counts are Poisson with

    log(lambda_home) = intercept + home_adv + attack[home] - defence[away]
    log(mu_away)     = intercept            + attack[away] - defence[home]

plus the Dixon-Coles low-score correction (rho) that fixes the Poisson
model's known miscalibration on 0-0/1-0/0-1/1-1 scorelines, and exponential
time-decay weighting so recent form counts more than old seasons.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # score grid is (MAX_GOALS+1) x (MAX_GOALS+1)

# Dixon & Coles estimated xi ~= 0.0065 per half-week on English data,
# which is about 0.0019 per day.
DEFAULT_XI = 0.0019


def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles dependence correction for low-scoring results."""
    tau = np.ones_like(lam)
    tau = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1 + lam * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1 + mu * rho, tau)
    tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
    return np.clip(tau, 1e-10, None)


class DixonColesModel:
    def __init__(self, xi: float = DEFAULT_XI):
        self.xi = xi
        self.teams: list[str] = []
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.home_adv: float = 0.0
        self.intercept: float = 0.0
        self.rho: float = 0.0

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColesModel":
        """Fit on matches played strictly before as_of (default: use everything)."""
        df = matches
        if as_of is not None:
            df = df[df["Date"] < as_of]
        if as_of is None:
            as_of = df["Date"].max() + pd.Timedelta(days=1)
        if len(df) < 50:
            raise ValueError(f"Only {len(df)} matches before {as_of.date()} — not enough to fit")

        self.teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = df["HomeTeam"].map(idx).to_numpy()
        ai = df["AwayTeam"].map(idx).to_numpy()
        hg = df["FTHG"].to_numpy(dtype=float)
        ag = df["FTAG"].to_numpy(dtype=float)
        days_ago = (as_of - df["Date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days_ago)

        def unpack(x):
            att, def_ = x[:n], x[n:2 * n]
            # Centre ratings so the intercept is identifiable.
            att = att - att.mean()
            def_ = def_ - def_.mean()
            return att, def_, x[2 * n], x[2 * n + 1], x[2 * n + 2]

        def nll(x):
            att, def_, home, intercept, rho = unpack(x)
            lam = np.exp(intercept + home + att[hi] - def_[ai])
            mu = np.exp(intercept + att[ai] - def_[hi])
            ll = (
                np.log(_tau(hg, ag, lam, mu, rho))
                + hg * np.log(lam) - lam
                + ag * np.log(mu) - mu
            )
            return -(w * ll).sum()

        x0 = np.concatenate([np.zeros(2 * n), [0.25, 0.1, 0.0]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-2, 2), (-0.2, 0.2)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        self.attack, self.defence, self.home_adv, self.intercept, self.rho = unpack(res.x)
        return self

    def _rates(self, home: str, away: str) -> tuple[float, float]:
        idx = {t: i for i, t in enumerate(self.teams)}
        for t in (home, away):
            if t not in idx:
                raise KeyError(f"Team {t!r} not in fitted model. Known teams: {self.teams}")
        h, a = idx[home], idx[away]
        lam = np.exp(self.intercept + self.home_adv + self.attack[h] - self.defence[a])
        mu = np.exp(self.intercept + self.attack[a] - self.defence[h])
        return float(lam), float(mu)

    def score_grid(self, home: str, away: str) -> np.ndarray:
        """Joint probability of every scoreline up to MAX_GOALS each."""
        lam, mu = self._rates(home, away)
        goals = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        grid *= _tau(hg.astype(float), ag.astype(float),
                     np.full_like(grid, lam), np.full_like(grid, mu), self.rho)
        return grid / grid.sum()

    def outcome_probs(self, home: str, away: str) -> dict[str, float]:
        """P(home win), P(draw), P(away win)."""
        grid = self.score_grid(home, away)
        return {
            "H": float(np.tril(grid, -1).sum()),
            "D": float(np.trace(grid)),
            "A": float(np.triu(grid, 1).sum()),
        }

    def ratings(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )
