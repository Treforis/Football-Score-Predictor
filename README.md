# Football Score Predictor

Machine learning project to predict outcomes in the 2026/27 football season across the top five European leagues.

A Streamlit app that browses fixtures by matchday, predicts home/draw/away probabilities and expected scorelines, and projects a final league table. Trained on 33 seasons of historical results; fixtures and live results pulled from the football-data.org API.

**Leagues covered:** La Liga, Premier League, Bundesliga, Serie A, Ligue 1

## Results

Log loss on a held-out test set (all matches from August 2023 onward), compared against a base-rate baseline that predicts each league's historical H/D/A frequencies.

| League | Baseline | Model | Improvement | Accuracy | Home MAE |
|---|---|---|---|---|---|
| Serie A | 1.0950 | 0.9932 | 0.1018 | 0.5216 | 0.907 |
| La Liga | 1.0651 | 0.9705 | 0.0946 | 0.5280 | 0.926 |
| Bundesliga | 1.0832 | 1.0089 | 0.0743 | 0.4923 | 1.082 |
| Premier League | 1.0763 | 1.0038 | 0.0725 | 0.5189 | 0.983 |
| Ligue 1 | 1.0809 | 1.0160 | 0.0649 | 0.4962 | 0.982 |

Uniform three-class guessing scores ln(3) = 1.099. The model improves on base rates by 0.065–0.102 log loss across all five leagues without any per-league tuning.

Raw log loss is a poor way to compare leagues against each other. Serie A looks mid-table on the raw number but extracts the most signal, because its baseline is the hardest of the five.

## How it works

### Features

Ten features per match, all computed from information available before kickoff:

```
home_gf_last5, home_ga_last5, home_pts_last5, home_days_rest,
away_gf_last5, away_ga_last5, away_pts_last5, away_days_rest,
home_elo, away_elo
```

Rolling averages use `.shift(1)` so the current match is never included in its own features.

### Elo ratings

Sequential loop over date-sorted matches. K = 20, starting rating 1500, +60 home advantage in the expected-score calculation. Ratings regress toward the mean at each season boundary:

```
new_rating = 1500 + (rating - 1500) * 0.75
```

The rating is recorded *before* each match, then updated. This is the leakage guard.

### Models

Three `HistGradientBoostingRegressor`/`Classifier` models per league: one classifier for the H/D/A outcome, two regressors for home and away goals. No feature scaling needed.

### Train/test split

Split by date at 2023-08-01. Never a random split, because a random split would let the model train on matches that happen after the ones it's tested on.

## Project structure

```
Football-Score-Predictor/
├── data/                    # training CSVs (gitignored, rebuild with download.py)
│   ├── LL/   SP1_1993.csv … SP1_2025.csv
│   ├── EPL/  E0_1993.csv  … E0_2025.csv
│   ├── BL1/  D1_1993.csv  … D1_2025.csv
│   ├── SA/   I1_1993.csv  … I1_2025.csv
│   └── FL1/  F1_1993.csv  … F1_2025.csv
├── artifacts/
│   ├── PD/  PL/  BL1/  SA/  FL1/
│   │   ├── model.pkl, model_home_goals.pkl, model_away_goals.pkl
│   │   ├── snapshot.csv, snapshot_live.csv
│   │   └── fixtures.csv, weekly.csv
│   └── last_update.txt
├── download.py              # fetch historical CSVs from football-data.co.uk
├── pipeline.py              # data → features → Elo → train → save artifacts
├── fetch.py                 # API → fixtures → predictions → weekly table
├── app.py                   # Streamlit UI
└── update.bat               # fetch → pipeline → fetch
```

## Setup

```bash
pip install -r requirements.txt
python download.py           # downloads 165 season CSVs (~25 MB)
```

Create a `.env` file with a free API token from [football-data.org](https://www.football-data.org/):

```
FOOTBALL_API_TOKEN=your_token_here
```

Then:

```bash
python pipeline.py           # train models, write artifacts
python fetch.py              # pull fixtures, predict, write tables
python -m streamlit run app.py
```

## Weekly update

```
update.bat
```

Runs `fetch.py` → `pipeline.py` → `fetch.py`. Order matters: fetch pulls finished results, pipeline retrains with them, fetch re-predicts using the updated ratings.

## The app

**Results.** Finished, live and upcoming matches, stepped through one day at a time, with club badges and real scores against predictions.

**Matchdays.** Fixture browser for any matchday. Played matches show the real score with the prediction underneath.

**Table.** Projected final standings. Uses actual points for played matches and expected points for unplayed ones, so the projection updates as the season progresses. Expected points are `3 × P(win) + P(draw)`, not summed argmax picks, because summing picks gives nonsense (Barcelona on 111 points).

## Notes and limitations

Promoted teams with no top-flight history get default features (Elo 1400, league-average form). Predictions involving them are confident but unfounded. Currently ten such teams across the five leagues.

Snapshot form features freeze at the end of the last completed season until `pipeline.py` reruns.

The free API tier allows 10 requests per minute and returns no in-play data, so scores appear only once a match is `FINISHED`.

## What was tested

| Change | Δ log loss | Verdict |
|---|---|---|
| Adding Elo | −0.048 | large |
| More history (EPL 1993 vs 2005) | −0.025 | real |
| Season regression (0.75) | −0.011 | real |
| Elo home advantage (+60) | −0.0004 | nothing |
| Logistic → gradient boosting | −0.0003 | nothing |
| Hyperparameter tuning | −0.0007 | nothing |
| 10- and 20-match rolling windows | +0.012 | worse, reverted |
| xG from FBref | none | nothing over Elo |

The pattern: only changes that add genuinely new information help. Rearranging what the model already knows does nothing. The feature set is the constraint, not the model.

## Data sources

Historical match results from [football-data.co.uk](https://www.football-data.co.uk/). Fixtures, live results and club badges from [football-data.org](https://www.football-data.org/).
