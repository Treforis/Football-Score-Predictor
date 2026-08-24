import os 
import time
import requests

LEAGUES = {'SP1': 'data/LL', 'E0': 'data/EPL', 'D1': 'data/BL1',
           'I1': 'data/SA', 'F1': 'data/FL1'}


for div, folder in LEAGUES.items():
    os.makedirs(folder, exist_ok=True)
    for year in range(1993, 2026):
        season = f"{str(year)[2:]}{str(year+1)[2:]}"
        out = f"{folder}/{div}_{year}.csv"

        if os.path.exists(out):
            continue

        url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

        r = requests.get(url)
        if r.status_code != 200 or len(r.content) < 1000:
            print(f"skip {div} {season} ({r.status_code})")
            continue

        with open(out, 'wb') as fh:
            fh.write(r.content)
        print(f"got {out} ({len(r.content)//1024} KB)")
        time.sleep(0.5)