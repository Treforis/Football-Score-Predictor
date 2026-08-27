"""
download.py — build the data/ folder from scratch.

data/ is gitignored, so run this once after cloning, before pipeline.py.
33 seasons x 5 leagues from football-data.co.uk, roughly 25 MB.
Safe to re-run: existing files are skipped.
"""

import os
import time
import requests

# football-data.co.uk division codes mapped to the folders pipeline.py globs.
LEAGUES = {'SP1': 'data/LL', 'E0': 'data/EPL', 'D1': 'data/BL1',
           'I1': 'data/SA', 'F1': 'data/FL1'}


for div, folder in LEAGUES.items():
    os.makedirs(folder, exist_ok=True)
    # Not every league goes back to 1993, so early misses are expected.
    for year in range(1993, 2026):
        season = f"{str(year)[2:]}{str(year+1)[2:]}"   # 1993-94 -> "9394"
        # Start year in the filename, because pipeline.py reads the season
        # from the name rather than the contents.
        out = f"{folder}/{div}_{year}.csv"

        # Delete a file to force a refresh of just that one.
        if os.path.exists(out):
            continue

        url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

        r = requests.get(url)
        # The size check catches the site returning an HTML error page with a
        # 200 status, which would save as a valid-looking but broken CSV.
        if r.status_code != 200 or len(r.content) < 1000:
            print(f"skip {div} {season} ({r.status_code})")
            continue

        # Bytes, not text: these files aren't utf-8. pipeline.py decodes them.
        with open(out, 'wb') as fh:
            fh.write(r.content)
        print(f"got {out} ({len(r.content)//1024} KB)")
        time.sleep(0.5)   # 165 rapid hits on a free public site risks a block