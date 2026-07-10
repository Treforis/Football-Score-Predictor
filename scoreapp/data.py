"""Historical match data from football-data.co.uk (free CSVs, top European leagues).

Files are laid out as https://www.football-data.co.uk/mmz4281/<season>/<league>.csv
where <season> is e.g. "2526" for 2025-26 and <league> is the division code.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "E0": "Premier League (England)",
    "SP1": "La Liga (Spain)",
    "D1": "Bundesliga (Germany)",
    "I1": "Serie A (Italy)",
    "F1": "Ligue 1 (France)",
}

# Friendly aliases accepted on the CLI.
ALIASES = {
    "epl": "E0", "premierleague": "E0", "premier-league": "E0",
    "laliga": "SP1", "la-liga": "SP1",
    "bundesliga": "D1",
    "seriea": "I1", "serie-a": "I1",
    "ligue1": "F1", "ligue-1": "F1",
}

CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLS = ["B365H", "B365D", "B365A"]  # bookmaker odds, used as a benchmark


def resolve_league(name: str) -> str:
    code = ALIASES.get(name.lower().replace(" ", ""), name.upper())
    if code not in LEAGUES:
        raise ValueError(f"Unknown league {name!r}. Known: {', '.join(LEAGUES)}")
    return code


def season_range(start: str, end: str) -> list[str]:
    """Expand "2122".."2526" into ["2122", "2223", ...]."""
    seasons = []
    year = int(start[:2])
    while True:
        code = f"{year:02d}{(year + 1) % 100:02d}"
        seasons.append(code)
        if code == end:
            return seasons
        year = (year + 1) % 100
        if len(seasons) > 60:
            raise ValueError(f"Bad season range {start}..{end}")


def parse_seasons(spec: str) -> list[str]:
    """Parse "2122-2526" or "2425,2526" into season codes."""
    if "-" in spec:
        start, end = spec.split("-")
        return season_range(start, end)
    return spec.split(",")


def fetch(league: str, seasons: list[str], data_dir: Path, refresh: bool = False) -> list[Path]:
    """Download season CSVs, skipping files already on disk unless refresh=True."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for season in seasons:
        dest = data_dir / f"{league}_{season}.csv"
        if refresh or not dest.exists():
            url = f"{BASE_URL}/{season}/{league}.csv"
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    dest.write_bytes(resp.read())
            except Exception as exc:  # noqa: BLE001 - report and continue with other seasons
                print(f"warning: could not fetch {url}: {exc}")
                continue
        paths.append(dest)
    return paths


def load(league: str, seasons: list[str], data_dir: Path) -> pd.DataFrame:
    """Load and concatenate season CSVs into one tidy match DataFrame."""
    frames = []
    for season in seasons:
        path = data_dir / f"{league}_{season}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run the fetch command first")
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        keep = CORE_COLS + [c for c in ODDS_COLS if c in df.columns]
        df = df[keep].copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        df["Season"] = season
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    out[["FTHG", "FTAG"]] = out[["FTHG", "FTAG"]].astype(int)
    return out
