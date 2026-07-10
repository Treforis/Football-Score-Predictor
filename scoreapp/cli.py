"""Command-line interface.

    python -m scoreapp fetch    --league epl --seasons 2122-2526
    python -m scoreapp predict  --league epl --home Arsenal --away Chelsea
    python -m scoreapp simulate --league epl --season 2526 --as-of 2026-01-01 --runs 10000
    python -m scoreapp backtest --league epl --season 2526
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import backtest as bt
from . import data
from .model import DixonColesModel
from .simulate import simulate_season

DEFAULT_SEASONS = "2122-2526"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--league", required=True, help="e.g. epl, laliga, bundesliga, seriea, ligue1 (or E0, SP1, D1, I1, F1)")
    p.add_argument("--seasons", default=DEFAULT_SEASONS, help="training data, e.g. 2122-2526")
    p.add_argument("--data-dir", default="data", type=Path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scoreapp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download historical CSVs")
    _add_common(p_fetch)
    p_fetch.add_argument("--refresh", action="store_true")

    p_pred = sub.add_parser("predict", help="predict one match")
    _add_common(p_pred)
    p_pred.add_argument("--home", required=True)
    p_pred.add_argument("--away", required=True)

    p_sim = sub.add_parser("simulate", help="Monte Carlo the rest of a season")
    _add_common(p_sim)
    p_sim.add_argument("--season", required=True, help="season to simulate, e.g. 2526")
    p_sim.add_argument("--as-of", required=True, help="cutoff date YYYY-MM-DD; matches after it are simulated")
    p_sim.add_argument("--runs", type=int, default=10_000)
    p_sim.add_argument("--relegation-slots", type=int, default=3)
    p_sim.add_argument("--seed", type=int, default=None)

    p_back = sub.add_parser("backtest", help="walk-forward evaluation of a season")
    _add_common(p_back)
    p_back.add_argument("--season", required=True)

    args = parser.parse_args(argv)
    league = data.resolve_league(args.league)
    seasons = data.parse_seasons(args.seasons)

    if args.cmd == "fetch":
        paths = data.fetch(league, seasons, args.data_dir, refresh=args.refresh)
        print(f"{len(paths)} file(s) in {args.data_dir}/ for {data.LEAGUES[league]}")
        return

    matches = data.load(league, seasons, args.data_dir)

    if args.cmd == "predict":
        model = DixonColesModel().fit(matches)
        probs = model.outcome_probs(args.home, args.away)
        grid = model.score_grid(args.home, args.away)
        top = sorted(
            ((h, a, grid[h, a]) for h in range(grid.shape[0]) for a in range(grid.shape[1])),
            key=lambda x: -x[2],
        )[:5]
        print(f"{args.home} vs {args.away}  ({data.LEAGUES[league]})")
        print(f"  home win {probs['H']:.1%}   draw {probs['D']:.1%}   away win {probs['A']:.1%}")
        print("  most likely scores: " + ", ".join(f"{h}-{a} ({p:.1%})" for h, a, p in top))

    elif args.cmd == "simulate":
        as_of = pd.Timestamp(args.as_of)
        season = matches[matches["Season"] == args.season]
        if season.empty:
            raise SystemExit(f"No matches for season {args.season} — check --seasons includes it")
        played = season[season["Date"] < as_of]
        fixtures = season[season["Date"] >= as_of]
        model = DixonColesModel().fit(matches, as_of=as_of)
        table = simulate_season(model, played, fixtures, runs=args.runs,
                                relegation_slots=args.relegation_slots, seed=args.seed)
        print(f"{data.LEAGUES[league]} {args.season}: {len(fixtures)} fixtures simulated "
              f"{args.runs:,} times from {as_of.date()}\n")
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(table.to_string(index=False))

    elif args.cmd == "backtest":
        preds = bt.walk_forward(matches, args.season)
        print(f"{data.LEAGUES[league]} {args.season}: walk-forward over {len(preds)} matches "
              "(weekly refits, no lookahead)\n")
        print(bt.summarise(preds).to_string())
        print("\nLower RPS/Brier/log-loss is better; bookmaker row is the bar to aim for.")
