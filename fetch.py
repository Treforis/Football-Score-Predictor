import os
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("FOOTBALL_API_TOKEN")

DEFAULT = {'gf_now': 1.0, 'ga_now': 1.6, 'pts_now': 1.0, 'elo': 1400}

NAME_MAPS = {
    'PD': {
        'Alavés': 'Alaves',
        'Athletic': 'Ath Bilbao',
        'Atleti': 'Ath Madrid',
        'Barça': 'Barcelona',
        'Espanyol': 'Espanol',
        'Rayo Vallecano': 'Vallecano',
        'Real Betis': 'Betis',
        'Real Sociedad': 'Sociedad',
        'Sevilla FC': 'Sevilla',
    },
    'PL': {
        'Arsenal FC': 'Arsenal',
        'Aston Villa FC': 'Aston Villa',
        'Chelsea FC': 'Chelsea',
        'Everton FC': 'Everton',
        'Fulham FC': 'Fulham',
        'Liverpool FC': 'Liverpool',
        'Man City': 'Man City',
        'Man United': 'Man United',
        'Newcastle': 'Newcastle',
        "Nott'm Forest": "Nott'm Forest",
        'Tottenham': 'Tottenham',
        'West Ham': 'West Ham',
        'Wolverhampton': 'Wolves',
        'Brighton Hove': 'Brighton',
        'Leeds United': 'Leeds',
        'Burnley FC': 'Burnley',
        'Sunderland AFC': 'Sunderland',
        'Brentford FC': 'Brentford',
        'Bournemouth': 'Bournemouth',
        'Crystal Palace FC': 'Crystal Palace',
        'Nottingham': "Nott'm Forest",
        'Leeds United FC': 'Leeds',
        'Ipswich Town': 'Ipswich',
        'Coventry City': 'Coventry',
        'Hull City': 'Hull',
    
    },
    'BL1': {
        '1. FC Köln': 'FC Koln',
        'Bayern': 'Bayern Munich',
        'Bremen': 'Werder Bremen',
        'Frankfurt': 'Ein Frankfurt',
        'HSV': 'Hamburg',
    },
    'SA': {
        'Como 1907': 'Como',
        'Venezia FC': 'Venezia',
    },
    'FL1': {
        'Olympique Lyon': 'Lyon',
        'PSG': 'Paris SG',
        'RC Lens': 'Lens',
        'Stade Rennais': 'Rennes',
        'Angers SCO': 'Angers',
    },
}


def fetch_league(code, season="2026"):
    r = requests.get(
        f"https://api.football-data.org/v4/competitions/{code}/matches",
        headers={"X-Auth-Token": TOKEN},
        params={"season": season},
    )

    if r.status_code != 200:
        print(f"{code}: API returned {r.status_code}")
        return

    rows = []
    for m in r.json()['matches']:
        rows.append({
            'date': m['utcDate'][:10],
            'matchday': m['matchday'],
            'home': m['homeTeam']['shortName'],
            'away': m['awayTeam']['shortName'],
            'status': m['status'],
            'home_goals': m['score']['fullTime']['home'],
            'away_goals': m['score']['fullTime']['away'],
            'date': m['utcDate'][:10],
            'utc': m['utcDate'],
            'home_crest': m['homeTeam']['crest'],
            'away_crest': m['awayTeam']['crest'],
        })

    fixtures = pd.DataFrame(rows)
    nm = NAME_MAPS.get(code, {})
    fixtures['home'] = fixtures['home'].replace(nm)
    fixtures['away'] = fixtures['away'].replace(nm)

    # actual results
    played = fixtures['status'] == 'FINISHED'
    fixtures['actual_ftr'] = np.select(
        [played & (fixtures['home_goals'] > fixtures['away_goals']),
         played & (fixtures['home_goals'] == fixtures['away_goals']),
         played & (fixtures['home_goals'] < fixtures['away_goals'])],
        ['H', 'D', 'A'], default=None)

    # snapshot + promoted teams
    snap = pd.read_csv(f"artifacts/{code}/snapshot.csv")
    missing = sorted((set(fixtures['home']) | set(fixtures['away'])) - set(snap['team']))
    print(f"{code} unmatched teams: {missing}")

    for t in missing:
        snap = pd.concat([snap, pd.DataFrame([{'team': t, **DEFAULT}])], ignore_index=True)

    # features
    snap_idx = snap.set_index('team')
    feat_rows = []
    for f in fixtures.itertuples():
        h = snap_idx.loc[f.home]
        a = snap_idx.loc[f.away]
        feat_rows.append({
            'home_gf_last5': h['gf_now'], 'home_ga_last5': h['ga_now'],
            'home_pts_last5': h['pts_now'], 'home_days_rest': 7,
            'away_gf_last5': a['gf_now'], 'away_ga_last5': a['ga_now'],
            'away_pts_last5': a['pts_now'], 'away_days_rest': 7,
            'home_elo': h['elo'], 'away_elo': a['elo'],
        })
    X = pd.DataFrame(feat_rows)

    # predictions
    model = joblib.load(f"artifacts/{code}/model.pkl")
    probs = model.predict_proba(X)
    for i, cls in enumerate(model.classes_):
        fixtures[f'p_{cls}'] = probs[:, i]
    fixtures['pred_ftr'] = model.classes_[probs.argmax(axis=1)]

    gh = joblib.load(f"artifacts/{code}/model_home_goals.pkl")
    ga = joblib.load(f"artifacts/{code}/model_away_goals.pkl")
    fixtures['xg_home'] = gh.predict(X).round(1)
    fixtures['xg_away'] = ga.predict(X).round(1)

    # weekly long table
    rows = []
    for f in fixtures.itertuples():
        rows.append({
            'matchday': f.matchday, 'team': f.home, 'opponent': f.away,
            'xW': f.p_H, 'xD': f.p_D, 'xL': f.p_A,
            'exp_pts': 3 * f.p_H + f.p_D,
            'pts': 3 if f.actual_ftr == 'H' else (1 if f.actual_ftr == 'D' else 0),
            'played': int(f.status == 'FINISHED'),
        })
        rows.append({
            'matchday': f.matchday, 'team': f.away, 'opponent': f.home,
            'xW': f.p_A, 'xD': f.p_D, 'xL': f.p_H,
            'exp_pts': 3 * f.p_A + f.p_D,
            'pts': 3 if f.actual_ftr == 'A' else (1 if f.actual_ftr == 'D' else 0),
            'played': int(f.status == 'FINISHED'),
        })
    weekly = pd.DataFrame(rows)

    # finished results back to CSV for retraining
    done = fixtures[fixtures['status'] == 'FINISHED']
    if len(done) > 0:
        pd.DataFrame({
            'Season': '2026-27',
            'Date': pd.to_datetime(done['date']),
            'HomeTeam': done['home'],
            'AwayTeam': done['away'],
            'FTHG': done['home_goals'].astype(int),
            'FTAG': done['away_goals'].astype(int),
            'FTR': done['actual_ftr'],
        }).to_csv(f"data/results_{code}_2026_27.csv", index=False)

    fixtures.to_csv(f"artifacts/{code}/fixtures.csv", index=False)
    weekly.to_csv(f"artifacts/{code}/weekly.csv", index=False)
    snap.to_csv(f"artifacts/{code}/snapshot_live.csv", index=False)

    print(f"{code}: {len(fixtures)} fixtures, {len(done)} played")


for code in ['PD', 'PL', 'BL1', 'SA', 'FL1']:
    fetch_league(code)

open("artifacts/last_update.txt", "w").write(str(datetime.now()))