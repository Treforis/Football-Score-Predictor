# ScoreApp — Design Document

A live football score app (in the vein of Sofascore / OneFootball) with a
machine-learning layer that predicts match outcomes and simulates entire
leagues and tournaments. Initial scope: the top 5 European leagues (Premier
League, La Liga, Bundesliga, Serie A, Ligue 1), designed to scale to more
competitions later.

---

## 1. Product vision

Two things existing score apps don't give users:

1. **Probabilities, not just scores.** Every match shows win/draw/loss
   probabilities and most-likely scorelines — before kickoff *and updating
   live as the match state changes* (score, red cards, time remaining).
2. **Season/tournament simulation.** "What are Arsenal's title chances
   *right now*, mid-matchweek?" Answered by Monte Carlo — simulating every
   remaining fixture thousands of times and aggregating the outcomes into
   title / top-4 / relegation probabilities that update live.

### MVP feature set
- Live scores, fixtures, tables for the top 5 leagues
- Pre-match win/draw/loss probabilities + most likely scorelines
- League simulator: P(title), P(top 4), P(relegation), expected points/position
- Match page with basic live stats (possession, shots, xG when available)

### v2+
- In-play probability updates (win probability chart during the match)
- Tournament mode (Champions League groups + knockout brackets, extra time/penalty modelling)
- Push notifications ("Arsenal's title probability just moved 8% on this goal")
- More leagues (Eredivisie, Primeira Liga, Championship…), cups, internationals
- Player-level stats and ratings

---

## 2. Data strategy

This is the most important commercial decision in the project. Sofascore-class
apps are built on Opta/StatsPerform data that costs serious money; the way to
start is a tiered approach:

| Source | What it gives you | Live? | Cost | Use for |
|---|---|---|---|---|
| **football-data.co.uk** | Final scores, shots, cards, closing odds; 25+ seasons, top-5 leagues | No | Free | Model training + backtesting (used by the prototype) |
| **API-Football (api-sports.io)** | Fixtures, live scores/events (~15s), lineups, standings, in-play stats | Yes | Free tier (100 req/day) → ~$25–40/mo | MVP live layer |
| **football-data.org** | Fixtures, scores, standings | Slow updates | Free tier | Backup/dev |
| **Sportmonks** | Everything above + richer live stats, better SLA | Yes | ~€39+/mo tiers | Growth stage |
| **Understat / FBref** | Shot-level xG, historical | No | Free (scrape, check ToS) | Feature enrichment for the v2 model |
| **StatsPerform (Opta) / Stats Bomb** | Full event data, official-grade | Yes | $$$ (enterprise) | Only when revenue justifies it |

**Recommendation:** train and backtest on football-data.co.uk (free, done in
this repo); run the MVP's live layer on API-Football's paid tier; add xG
features from Understat. Do **not** scrape Sofascore itself — it violates
their ToS and their internal API changes without notice.

Key principle: **decouple ingestion from everything else.** All providers are
normalised into our own schema at the ingestion boundary, so swapping
API-Football for Sportmonks (or adding Opta later) touches one adapter, not
the app.

---

## 3. System architecture

```
                        ┌────────────────────────────────────────────┐
                        │                Data providers               │
                        │  API-Football (live)  football-data.co.uk  │
                        └──────┬─────────────────────┬───────────────┘
                               │ poll 15–60s          │ nightly batch
                        ┌──────▼──────────┐   ┌──────▼──────────┐
                        │ Live ingestion   │   │ Batch ingestion │
                        │ worker (adapter  │   │ + training jobs │
                        │ per provider)    │   └──────┬──────────┘
                        └──────┬──────────┘          │
                               │ normalised events    │
                  ┌────────────▼────────────┐  ┌─────▼──────────────┐
                  │ PostgreSQL              │  │ Model registry      │
                  │ (matches, events,       │  │ (versioned fitted   │
                  │  stats, standings)      │  │  models + metrics)  │
                  └──────┬─────────┬────────┘  └─────┬──────────────┘
                         │         │                  │
              ┌──────────▼──┐   ┌──▼──────────────────▼───┐
              │ Redis        │   │ Prediction service       │
              │ cache +      │◄──│ (match probs, live       │
              │ pub/sub      │   │  updates) + Simulation   │
              └──────┬──────┘   │  engine (Monte Carlo)     │
                     │          └──────────────────────────┘
              ┌──────▼──────────────────────┐
              │ API gateway (FastAPI)        │
              │ REST + WebSocket             │
              └──────┬──────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Clients: web (Next.js),   │
        │ mobile (React Native)     │
        └──────────────────────────┘
```

### Components

- **Ingestion workers** (Python): poll the live provider every 15–60s during
  match windows (schedule known from fixtures, so you only poll while games
  are on — this keeps API costs down). Normalise into our schema, write to
  Postgres, publish deltas on Redis pub/sub.
- **PostgreSQL** as the single source of truth: `competitions`, `seasons`,
  `teams`, `matches`, `match_events` (goals/cards/subs, append-only),
  `match_stats` (snapshots of possession/shots/xG), `predictions` (every
  prediction we ever serve, stamped with model version — this is your future
  evaluation dataset), `simulations`.
- **Prediction service**: loads the current model from the registry, serves
  pre-match probabilities (cached — they only change when the model refits),
  and recomputes in-play probabilities on each event/state change.
- **Simulation engine**: the Monte Carlo worker. Triggered nightly, on final
  whistles, and on demand. 10,000 season runs is <2s with the vectorised
  numpy implementation in this repo, so "run it a multitude of times" is
  cheap — see §5 for how many runs you actually need.
- **API gateway**: REST for fixtures/tables/predictions, WebSocket (or SSE)
  for live score + probability pushes.
- **Clients**: Next.js web first (fastest to iterate, indexable), React
  Native later sharing the API.

Everything above runs comfortably on a single small VPS + managed Postgres
for the MVP; the components are separated so each can be scaled or replaced
independently later.

---

## 4. Machine learning design

Three model levels, each shipping on top of the last:

### Level 1 — Dixon-Coles baseline (implemented in this repo)
Poisson goal model with per-team attack/defence ratings, home advantage, a
low-score dependence correction, and exponential time-decay so recent form
outweighs old seasons. Fits in ~1s per league, needs only final scores, and
produces a full scoreline distribution — which is exactly what the Monte
Carlo engine needs to sample from.

Backtested walk-forward (weekly refits, no lookahead) on the 2025-26 Premier
League: **RPS 0.2097 vs 0.2054 for Bet365's own odds** — within 2% of the
bookmaker using nothing but final scores. That's the correct baseline to
build from.

### Level 2 — Feature-based model (pre-match)
Gradient boosting (LightGBM/XGBoost) predicting the same H/D/A distribution
from richer features: rolling xG for/against, Elo, days of rest, distance
travelled, lineup strength (are key players starting?), league position
pressure, referee, weather. Train on 10+ seasons; **calibrate** the outputs
(isotonic regression) so probabilities mean what they say. Keep Dixon-Coles
as the fallback and sanity check — a feature model that can't beat it in
walk-forward RPS doesn't ship.

### Level 3 — In-play model
Re-estimate the remaining-goals rates conditional on live state: current
score, minute, red cards, live xG accumulated. A clean formulation: the
pre-match goal rates decay over the remaining minutes, adjusted by a model
trained on historical in-play states (minute-bucketed gradient boosting, or
a state-space/Poisson-race model). Every live event triggers a re-predict →
WebSocket push → and feeds updated match probabilities into a fresh league
simulation, so title odds move *during* matches.

### Evaluation — "make sure I have a good score"
Two different things both matter, and the repo implements both:

1. **Prediction quality** → walk-forward backtesting (`scoreapp backtest`).
   Metrics: **RPS** (the standard for ordered 1X2 outcomes), Brier,
   log-loss, plus calibration curves. Benchmark every model against
   bookmaker closing odds from the same CSVs — the closing line is the
   strongest publicly available predictor, so matching it is excellent and
   beating it is exceptional. Never evaluate on data the model trained on.
2. **Simulation precision** → number of Monte Carlo runs (§5).

Rule of the road: every model change must show its walk-forward RPS on held
out seasons before deployment, and every served prediction is logged with
its model version so live performance is continuously measurable.

---

## 5. Monte Carlo simulation engine

For each remaining fixture, the match model gives a full scoreline
probability grid. One simulation run samples a score for every remaining
fixture, applies points/GD/GF tie-breaks, and records the final table. Doing
this N times yields each team's distribution over final positions.

- **How many runs?** Standard error of a probability estimate is
  `sqrt(p(1-p)/N)`. At N=10,000 a 50% probability is accurate to ±0.5pp,
  and at N=100,000 to ±0.16pp. The vectorised engine in this repo does
  10,000 runs of a half-season in under 2 seconds, so re-running constantly
  (after every final whistle, or live during matches) is trivial. Fix the
  RNG seed for reproducible published numbers.
- **Live mode:** during a matchday, replace in-progress fixtures' pre-match
  grids with the in-play model's conditional distributions and re-simulate —
  title probabilities tick in near-real time.
- **Tournament mode (v2):** same engine, different structure — sample group
  stages into standings, apply draw/seeding rules, then simulate knockout
  ties (two legs/away goals where applicable, extra time and a penalty
  shootout model, ~78% base rate is roughly a coin flip weighted by team
  strength).
- **Known simplification:** ties are broken points → GD → GF; La Liga and
  Serie A head-to-head rules are a v2 refinement.

---

## 6. Tech stack summary

| Layer | Choice | Why |
|---|---|---|
| ML / simulation | Python, numpy/scipy/pandas (+ LightGBM at level 2) | Ecosystem, speed of iteration; vectorised MC is fast enough |
| Backend API | FastAPI | Async, WebSockets, Pydantic schemas shared with ingestion |
| DB | PostgreSQL (+ Redis for cache/pub-sub) | Relational fits fixtures/tables; append-only events scale fine |
| Web | Next.js + TypeScript | SSR for SEO on match pages, fast iteration |
| Mobile | React Native (later) | One codebase, shares the API |
| Infra | Docker Compose → single VPS; managed Postgres | MVP-appropriate; split services only when load demands |

---

## 7. Roadmap

- **Phase 0 (this repo, done):** Dixon-Coles model + Monte Carlo simulator +
  walk-forward backtesting CLI over the top 5 leagues on free historical data.
- **Phase 1:** Postgres schema + API-Football ingestion (fixtures, results,
  standings); nightly refit + simulation; FastAPI read endpoints. *You now
  have a headless prediction product.*
- **Phase 2:** Next.js web app — scores, tables, match pages, probability
  displays, simulator page.
- **Phase 3:** Live: in-match polling, WebSocket pushes, level-2 model with
  xG features, live-updating league simulations.
- **Phase 4:** In-play win probability model, tournament mode (UCL), React
  Native app, more leagues.

---

## 8. Prototype in this repo

```
scoreapp/
  data.py       # football-data.co.uk download + normalisation (5 leagues)
  model.py      # time-weighted Dixon-Coles with low-score correction
  simulate.py   # vectorised Monte Carlo season simulator
  backtest.py   # walk-forward evaluation: RPS/Brier/log-loss vs Bet365
  cli.py        # fetch / predict / simulate / backtest commands
```

See README.md for setup and example runs, including the 2025-26 Premier
League simulated from New Year's Day and the backtest against the bookmaker.
