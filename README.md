# ScoreApp

Football score app with a machine-learning prediction layer — live scores in
the style of Sofascore/OneFootball, plus win probabilities and Monte Carlo
league/tournament simulation for the top 5 European leagues.

**Start with [DESIGN.md](DESIGN.md)** for the full product and system design.
This repo currently contains Phase 0: a working prediction + simulation
engine you can run today on free historical data.

## Setup

Python 3.11+:

```bash
python3 -m venv .venv           # or: uv venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Leagues: `epl`, `laliga`, `bundesliga`, `seriea`, `ligue1`.

```bash
# 1. Download free historical data (football-data.co.uk), seasons 2021-22 → 2025-26
python -m scoreapp fetch --league epl

# 2. Predict a single match
python -m scoreapp predict --league epl --home Arsenal --away Chelsea
#   home win 62.5%   draw 23.0%   away win 14.5%
#   most likely scores: 2-0 (11.8%), 1-0 (11.4%), 1-1 (11.0%), ...

# 3. Monte Carlo a season: simulate every fixture after the cutoff 10,000x
python -m scoreapp simulate --league epl --season 2526 --as-of 2026-01-01 --runs 10000
#   → per-team P(title), P(top 4), P(relegation), expected points/position

# 4. Backtest: walk-forward over a season (weekly refits, no lookahead),
#    scored against Bet365's own odds as the benchmark
python -m scoreapp backtest --league epl --season 2526
#                     rps    brier  log_loss  accuracy
#   model           0.2097  0.6183   1.0378    0.4749
#   bookmaker(B365) 0.2054  0.6111   1.0180    0.4908
```

The model is a time-decay-weighted Dixon-Coles Poisson model (attack/defence
ratings per team, home advantage, low-score correction) — the standard
academic baseline for football score prediction. On the 2025-26 Premier
League it lands within 2% of the bookmaker's ranked probability score using
nothing but final scores.
