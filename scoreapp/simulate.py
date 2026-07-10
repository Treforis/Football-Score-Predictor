"""Monte Carlo season simulation.

Take the table as it stands at a cutoff date, then play out every remaining
fixture thousands of times by sampling scorelines from the fitted model.
Aggregating the simulated final tables gives the probability of each team
winning the title, finishing top 4, or being relegated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import MAX_GOALS, DixonColesModel


def _table_from_matches(matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    idx = {t: i for i, t in enumerate(teams)}
    pts = np.zeros(len(teams))
    gf = np.zeros(len(teams))
    ga = np.zeros(len(teams))
    played = np.zeros(len(teams), dtype=int)
    for _, m in matches.iterrows():
        h, a = idx[m["HomeTeam"]], idx[m["AwayTeam"]]
        hg, ag = m["FTHG"], m["FTAG"]
        gf[h] += hg; ga[h] += ag
        gf[a] += ag; ga[a] += hg
        played[h] += 1; played[a] += 1
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    return pd.DataFrame({"team": teams, "played": played, "points": pts, "gf": gf, "ga": ga})


def simulate_season(
    model: DixonColesModel,
    played: pd.DataFrame,
    fixtures: pd.DataFrame,
    runs: int = 10_000,
    relegation_slots: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate the remaining fixtures `runs` times and summarise final positions.

    played:   completed matches of the season (with FTHG/FTAG).
    fixtures: remaining matches (HomeTeam/AwayTeam only are used).

    Ties are broken by goal difference then goals for (head-to-head rules used
    by some leagues are not modelled).
    """
    teams = sorted(
        set(played["HomeTeam"]) | set(played["AwayTeam"])
        | set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    )
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rng = np.random.default_rng(seed)

    base = _table_from_matches(played, teams)
    pts = np.tile(base["points"].to_numpy(), (runs, 1))
    gf = np.tile(base["gf"].to_numpy(), (runs, 1))
    ga = np.tile(base["ga"].to_numpy(), (runs, 1))

    side = MAX_GOALS + 1
    for _, m in fixtures.iterrows():
        h, a = idx[m["HomeTeam"]], idx[m["AwayTeam"]]
        grid = model.score_grid(m["HomeTeam"], m["AwayTeam"]).ravel()
        draw = rng.choice(grid.size, size=runs, p=grid)
        hg, ag = draw // side, draw % side
        gf[:, h] += hg; ga[:, h] += ag
        gf[:, a] += ag; ga[:, a] += hg
        home_win = hg > ag
        away_win = hg < ag
        tie = ~home_win & ~away_win
        pts[:, h] += 3 * home_win + tie
        pts[:, a] += 3 * away_win + tie

    # Rank teams inside each simulation: points, then GD, then GF.
    gd = gf - ga
    key = pts * 1e6 + gd * 1e3 + gf  # composite sort key, safe for realistic magnitudes
    order = np.argsort(-key, axis=1, kind="stable")
    position = np.empty_like(order)
    rows = np.arange(runs)[:, None]
    position[rows, order] = np.arange(1, n + 1)

    summary = pd.DataFrame({
        "team": teams,
        "played": base["played"],
        "points_now": base["points"].astype(int),
        "exp_points": pts.mean(axis=0).round(1),
        "exp_position": position.mean(axis=0).round(2),
        "p_title": (position == 1).mean(axis=0),
        "p_top4": (position <= 4).mean(axis=0),
        "p_relegation": (position > n - relegation_slots).mean(axis=0),
    })
    return summary.sort_values("exp_position").reset_index(drop=True)
