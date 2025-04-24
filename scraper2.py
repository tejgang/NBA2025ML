import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def get_team_abbreviations() -> list:
    """
    Scrape the teams index page to get all 3-letter team abbreviations,
    both active and inactive franchises.
    """
    url = "https://www.basketball-reference.com/teams/"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    abbrs = set()
    for table_id in ("teams_active", "teams_inactive"):
        tbl = soup.find("table", id=table_id)
        if not tbl:
            continue
        for a in tbl.select("tbody a[href^='/teams/']"):
            abbr = a["href"].split("/")[2]
            abbrs.add(abbr)
    return sorted(abbrs)

def get_seed_and_name(abbr: str, year: int, cache: dict) -> dict:
    """
    Fetch full team name and seed for a given season-year.
    Caches results in `cache` to avoid repeat HTTP calls.
    """
    key = f"{abbr}_{year}"
    if key in cache:
        return cache[key]

    url = f"https://www.basketball-reference.com/teams/{abbr}/{year}.html"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        cache[key] = {"name": None, "seed": None}
        return cache[key]

    soup = BeautifulSoup(r.content, "html.parser")

    # Extract name from H1 ("2006 Miami Heat" → "Miami Heat")
    h1 = soup.find("h1").get_text(strip=True)
    name = re.sub(r"^\d{4}\s+", "", h1)

    # Find text like "52-30, 2nd in East" to get seed
    rec = soup.find(text=re.compile(r"in (East|West|Atlantic|Central|Pacific|Southeast|Northwest|Southwest)"))
    seed = None
    if rec:
        m = re.search(r"(\d+)(?:st|nd|rd|th) in", rec)
        if m:
            seed = int(m.group(1))

    cache[key] = {"name": name, "seed": seed}
    time.sleep(0.2)
    return cache[key]

def scrape_playoff_games_year(year: int, team_abbrs: list, cache: dict) -> list:
    """
    For a given year, iterate every team abbreviation in team_abbrs,
    scrape that team's game-log page, pull out playoff home games,
    and return a list of game-record dicts.
    """
    games = []
    for abbr in team_abbrs:
        # Fetch the team's game log page for that year
        url = f"https://www.basketball-reference.com/teams/{abbr}/{year}_games.html"
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            continue

        soup = BeautifulSoup(r.content, "html.parser")
        # Un-comment any hidden tables
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()

        # Locate the playoffs table by id starting with 'playoffs'
        playoff_table = None
        for tbl in soup.find_all("table"):
            tid = tbl.get("id", "").lower()
            if tid.startswith("playoffs"):
                playoff_table = tbl
                break
        if playoff_table is None:
            continue  # this team didn't make playoffs or structure changed

        # Get home-team info
        info = get_seed_and_name(abbr, year, cache)
        team1_name = info["name"]
        seed_home  = info["seed"]

        # Iterate over rows in the playoffs table
        for row in playoff_table.find("tbody").find_all("tr"):
            # skip header rows
            if row.get("class") and "thead" in row["class"]:
                continue

            # Ensure it's marked Playoffs
            gt = row.find("td", {"data-stat": "game_type"})
            if not gt or gt.text.strip() != "Playoffs":
                continue

            # Only home games: location cell blank (away games marked '@')
            loc = row.find("td", {"data-stat": "game_location"}).text.strip()
            if loc != "":
                continue

            # Opponent
            opp_cell = row.find("td", {"data-stat": "opp_name"})
            if not opp_cell or not opp_cell.find("a"):
                continue
            opp_abbr  = opp_cell.find("a")["href"].split("/")[2]
            team2_name = opp_cell.get_text(strip=True)

            # Scores
            try:
                pts_h = int(row.find("td", {"data-stat": "pts"}).text)
                pts_a = int(row.find("td", {"data-stat": "opp_pts"}).text)
            except:
                continue

            # Visitor seed & name
            opp_info = get_seed_and_name(opp_abbr, year, cache)
            seed_away = opp_info["seed"]

            games.append({
                "team1":      team1_name,
                "team2":      team2_name,
                "team1_id":   abbr,
                "team2_id":   opp_abbr,
                "team1_seed": seed_home,
                "team2_seed": seed_away,
                "team1_win":  pts_h > pts_a,
                "year":       year
            })

        # be polite
        time.sleep(0.2)

    return games

def compile_all_playoff_games(start_year=2001, end_year=2025) -> pd.DataFrame:
    team_abbrs = get_team_abbreviations()
    cache = {}
    all_games = []

    for yr in range(start_year, end_year + 1):
        print(f"→ Scraping {yr} playoffs…")
        year_games = scrape_playoff_games_year(yr, team_abbrs, cache)
        all_games.extend(year_games)

    df = pd.DataFrame(all_games)
    # Ensure correct column order
    cols = [
        "team1","team2",
        "team1_id","team2_id",
        "team1_seed","team2_seed",
        "team1_win","year"
    ]
    # Only keep columns that exist
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv("game_by_game_2001_2025.csv", index=False)
    return df

if __name__ == "__main__":
    df = compile_all_playoff_games(2001, 2025)
    print(df.head())
