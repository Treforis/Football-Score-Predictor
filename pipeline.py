"""
pipeline.py — train the models.

Per league: load season CSVs -> Elo -> rolling form features -> split by
date -> train three models -> save artifacts. Everything lives inside
build(), so leagues can't contaminate each other.
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

FEATURES = ['home_gf_last5', 'home_ga_last5', 'home_pts_last5', 'home_days_rest',
            'away_gf_last5', 'away_ga_last5', 'away_pts_last5', 'away_days_rest',
            'home_elo', 'away_elo']

ROLL = ['Date', 'team', 'gf_last5', 'ga_last5', 'pts_last5', 'days_rest']

# Keys are the API's league codes (used for artifact folders); the file
# patterns are football-data.co.uk's division codes. The two are unrelated.
LEAGUES = {
    'PD':  {'dir': 'data/LL',   'pattern': 'SP1_*.csv'},
    'PL':  {'dir': 'data/EPL',  'pattern': 'E0_*.csv'},
    'BL1': {'dir': 'data/BL1',  'pattern': 'D1_*.csv'},
    'SA':  {'dir': 'data/SA',   'pattern': 'I1_*.csv'},
    'FL1': {'dir': 'data/FL1',  'pattern': 'F1_*.csv'},
}


def load_league(code):
    """Every historical match for one league, plus current-season results
    written back by fetch.py. Season comes from the filename."""
    cfg = LEAGUES[code]
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    frames = []
    files = sorted(glob.glob(f"{cfg['dir']}/{cfg['pattern']}"))

    # Fail loudly: an empty glob otherwise trains on almost nothing and
    # surfaces much later as a confusing sklearn error.
    if not files:
        raise FileNotFoundError(f"{code}: no files matching {cfg['dir']}/{cfg['pattern']}")

    for f in files:
        # latin-1: older files contain Windows-1252 bytes that crash a utf-8 read.
        d = pd.read_csv(f, usecols=cols, encoding='latin-1')
        year = int(f.split('_')[1][:4])
        d['Season'] = f"{year}-{str(year+1)[2:]}"
        frames.append(d)

    live = f"data/results_{code}_2026_27.csv"
    if os.path.exists(live):
        frames.append(pd.read_csv(live, encoding='latin-1'))

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['FTR', 'FTHG', 'FTAG'])
    # format='mixed': the source switched from dd/mm/yy to dd/mm/yyyy in 2017-18.
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    print(f"{code}: {len(df)} rows, {df['Season'].min()} -> {df['Season'].max()}")
    return df


def build(code, df):
    """Train and save the three models for one league."""
    df = df.sort_values('Date').reset_index(drop=True)

    # --- Elo ---
    # Must run on date-sorted rows, before the reshape.
    K = 20
    ratings = {}
    home_elos, away_elos = [], []
    current_season = None

    for row in df.itertuples():
        if row.Season != current_season:
            if current_season is not None:
                # Squad turnover means last season's gaps overstate this
                # season's. 0.75 tested best (0.6 -> 0.9731, 0.9 -> 0.9666).
                for t in ratings:
                    ratings[t] = 1500 + (ratings[t] - 1500) * 0.75
            current_season = row.Season

        # Record ratings BEFORE the result is applied. This is the leakage
        # guard: a row can never see the outcome of its own match.
        rh = ratings.get(row.HomeTeam, 1500)
        ra = ratings.get(row.AwayTeam, 1500)
        home_elos.append(rh)
        away_elos.append(ra)

        e_home = 1 / (1 + 10 ** ((ra - (rh + 60)) / 400))   # +60 is home advantage
        actual = {'H': 1.0, 'D': 0.5, 'A': 0.0}[row.FTR]
        change = K * (actual - e_home)

        ratings[row.HomeTeam] = rh + change
        ratings[row.AwayTeam] = ra - change

    df['home_elo'] = home_elos
    df['away_elo'] = away_elos

    # --- long reshape ---
    # One row per team per match, so a rolling window covers a team's last
    # five games whether they were home or away.
    df1 = df[['Date', 'HomeTeam', 'FTHG', 'FTAG']].copy()
    df1 = df1.rename(columns={'HomeTeam': 'team', 'FTHG': 'goals_for', 'FTAG': 'goals_against'})

    df2 = df[['Date', 'AwayTeam', 'FTAG', 'FTHG']].copy()
    df2 = df2.rename(columns={'AwayTeam': 'team', 'FTAG': 'goals_for', 'FTHG': 'goals_against'})

    long = pd.concat([df1, df2])
    long = long.sort_values(['team', 'Date']).reset_index(drop=True)

    long['points'] = np.select(
        [long['goals_for'] > long['goals_against'],
         long['goals_for'] == long['goals_against']],
        [3, 1], default=0)

    # _last5 is shifted, for training: a match is excluded from its own features.
    # _now is unshifted, for the snapshot only: the next fixture legitimately
    # gets to use the last completed match. Swapping them would be leakage.
    # Window 5 beat 10 and 20 (+0.012 log loss).
    for src, name in [('goals_for', 'gf'), ('goals_against', 'ga'), ('points', 'pts')]:
        long[f'{name}_last5'] = long.groupby('team')[src].transform(
            lambda x: x.shift(1).rolling(5).mean())
        long[f'{name}_now'] = long.groupby('team')[src].transform(
            lambda x: x.rolling(5).mean())

    long['days_rest'] = long.groupby('team')['Date'].transform(lambda x: x - x.shift(1)).dt.days

    # --- merge back ---
    for side, teamcol in [('home', 'HomeTeam'), ('away', 'AwayTeam')]:
        df = df.merge(long[ROLL], left_on=['Date', teamcol],
                      right_on=['Date', 'team'], how='left')
        df = df.rename(columns={c: f'{side}_{c}' for c in
                                ['gf_last5', 'ga_last5', 'pts_last5', 'days_rest']})
        df = df.drop(columns=['team'])

    # --- split ---
    model_df = df.dropna(subset=FEATURES).copy()

    # By date, never train_test_split. A random split would train on matches
    # that happen after the ones it's tested on.
    train = model_df[model_df['Date'] < '2023-08-01']
    test = model_df[model_df['Date'] >= '2023-08-01']

    X_train, y_train = train[FEATURES], train['FTR']
    X_test, y_test = test[FEATURES], test['FTR']

    # Predicting the league's H/D/A base rates and ignoring who's playing.
    # Raw log loss isn't comparable across leagues; the gap to this is.
    # labels= is required, or log_loss can line the columns up wrong.
    base = y_train.value_counts(normalize=True)
    base_probs = np.tile([base.get(c, 0) for c in ['A', 'D', 'H']], (len(y_test), 1))
    print(f"  baseline logloss {log_loss(y_test, base_probs, labels=['A','D','H']):.4f}"
          f"  home rate {(y_test == 'H').mean():.4f}")

    # --- models ---
    # Gradient boosting handles unscaled features, so there's no scaler here.
    # It also barely beat logistic regression (0.0003).
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.02,
                                        max_leaf_nodes=15, l2_regularization=1.0,
                                        random_state=42)
    gb.fit(X_train, y_train)

    gh = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)
    gh.fit(X_train, train['FTHG'])

    ga = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)
    ga.fit(X_train, train['FTAG'])

    # --- snapshot ---
    # Each current team's latest form and rating, for fetch.py to predict from.
    latest = long.sort_values('Date').groupby('team').tail(1)

    # Last TWO seasons, not max(). Once 2026-27 results merge in, max() returns
    # only the teams that have already played, and everyone else drops out of
    # the snapshot and falls through to fetch.py's DEFAULT features.
    seasons = sorted(df['Season'].unique())
    recent = df[df['Season'].isin(seasons[-2:])]
    current_teams = set(recent['HomeTeam']) | set(recent['AwayTeam'])

    snap = latest[latest['team'].isin(current_teams)].copy()
    snap['elo'] = snap['team'].map(ratings)

    # --- save ---
    os.makedirs(f"artifacts/{code}", exist_ok=True)
    joblib.dump(gb, f"artifacts/{code}/model.pkl")
    joblib.dump(gh, f"artifacts/{code}/model_home_goals.pkl")
    joblib.dump(ga, f"artifacts/{code}/model_away_goals.pkl")
    snap.to_csv(f"artifacts/{code}/snapshot.csv", index=False)

    # Log loss over accuracy: it scores confidence, not just the pick.
    probs = gb.predict_proba(X_test)
    print(f"\n{code}: {len(df)} matches, {len(train)} train, {len(test)} test")
    print(f"  logloss {log_loss(y_test, probs):.4f}  acc {accuracy_score(y_test, gb.predict(X_test)):.4f}")
    print(f"  home MAE {mean_absolute_error(test['FTHG'], gh.predict(X_test)):.4f}")
    # Sanity check: wrong teams at the top means a bad name mapping or join.
    print(pd.Series(ratings).sort_values(ascending=False).head(5))


for code in LEAGUES:
    build(code, load_league(code))