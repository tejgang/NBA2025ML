import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def get_seed_and_name(abbr: str, year: int, cache: dict) -> dict:
    """
    Returns {'name': full team name, 'seed': playoff seed} for team `abbr` in season `year`.
    Caches results in `cache` to avoid repeat requests.
    """
    key = f"{abbr}_{year}"
    if key in cache:
        return cache[key]

    # 1) Fetch team‐season page to get full name and regular‐season finish
    team_url = f"https://www.basketball-reference.com/teams/{abbr}/{year}.html"
    r = requests.get(team_url, headers=HEADERS)
    soup = BeautifulSoup(r.content, "html.parser")
    h1 = soup.find("h1").get_text(strip=True)       # e.g. "2006 Miami Heat"
    name = re.sub(r"^\d{4}\s+", "", h1)              # -> "Miami Heat"

    # 2) Parse seed from e.g. "52-30, 2nd in East" or "52-30, 2nd in Central"
    rec = soup.find(text=re.compile(r"in (East|West|Atlantic|Central|Pacific|Southeast|Northwest|Southwest)"))
    seed = None
    if rec:
        m = re.search(r"(\d+)(?:st|nd|rd|th) in", rec)
        if m:
            seed = int(m.group(1))

    cache[key] = {"name": name, "seed": seed}
    time.sleep(0.2)
    return cache[key]

def scrape_playoff_games_year(year: int, cache: dict) -> list:
    """
    Returns one dict per *single* playoff game in `year`, where:
      - team1 = home team
      - team2 = visitor
      - team1_id, team2_id = their 3-letter BBRef IDs
      - team1_seed, team2_seed
      - team1_win = True if home team won
      - year
    """
    # 1) Find which teams made the playoffs by looking at the bracket page
    bracket_url = f"https://www.basketball-reference.com/playoffs/NBA_{year}.html"
    r = requests.get(bracket_url, headers=HEADERS)
    soup = BeautifulSoup(r.content, "html.parser")

    # playoff teams appear in <a href="/teams/XXX/...">
    playoff_abbrs = {
        a["href"].split("/")[2]
        for a in soup.select("div.series_summary a[href^=\"/teams/\"]")
    }

    games = []
    for abbr in playoff_abbrs:
        info = get_seed_and_name(abbr, year, cache)
        full_name = info["name"]
        seed_home = info["seed"]

        # 2) Fetch the team's full game log page for that season
        games_url = f"https://www.basketball-reference.com/teams/{abbr}/{year}_games.html"
        gr = requests.get(games_url, headers=HEADERS)
        gsoup = BeautifulSoup(gr.content, "html.parser")

        # Remove commented-out tables (<-- ... -->)
        for c in gsoup.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()

        # Find the Playoffs table (id contains 'playoffs')
        playoff_table = None
        for tbl in gsoup.find_all("table"):
            if tbl.get("id", "").lower().startswith("playoffs"):
                playoff_table = tbl
                break
        if playoff_table is None:
            continue

        # 3) Iterate over each row of that table
        for row in playoff_table.tbody.find_all("tr"):
            # skip header sub-rows
            if row.get("class") and "thead" in row["class"]:
                continue

            # Confirm this is a Playoffs row
            gt = row.find("td", {"data-stat": "game_type"})
            if not gt or gt.text.strip() != "Playoffs":
                continue

            # Only take home games (game_location is blank for home; '@' for away)
            loc = row.find("td", {"data-stat": "game_location"}).text.strip()
            if loc != "":
                continue  # skip away

            # Opponent cell
            opp_cell = row.find("td", {"data-stat": "opp_name"})
            opp_abbr = opp_cell.find("a")["href"].split("/")[2]
            opp_name = opp_cell.text.strip()

            # Score cells
            pts_home = row.find("td", {"data-stat": "pts"}).text
            pts_away = row.find("td", {"data-stat": "opp_pts"}).text
            try:
                pts_h = int(pts_home)
                pts_a = int(pts_away)
            except ValueError:
                continue

            # Fetch visitor seed & name (caches automatically)
            opp_info = get_seed_and_name(opp_abbr, year, cache)
            seed_away = opp_info["seed"]

            games.append({
                "team1":      full_name,
                "team2":      opp_name,
                "team1_id":   abbr,
                "team2_id":   opp_abbr,
                "team1_seed": seed_home,
                "team2_seed": seed_away,
                "team1_win":  pts_h > pts_a,
                "year":       year
            })

        # be polite
        time.sleep(0.3)

    return games

def compile_all_playoff_games(start_year=2001, end_year=2025) -> pd.DataFrame:
    cache = {}
    all_games = []
    for yr in range(start_year, end_year + 1):
        print(f"Scraping playoffs {yr}…")
        try:
            season = scrape_playoff_games_year(yr, cache)
            all_games.extend(season)
        except Exception as e:
            print(f"  ⚠️  failed {yr}: {e}")
    df = pd.DataFrame(all_games)
    # ensure column order
    df = df[[
        "team1","team2",
        "team1_id","team2_id",
        "team1_seed","team2_seed",
        "team1_win","year"
    ]]
    # save to CSV
    df.to_csv("game_by_game_2001_2025.csv", index=False)
    return df

if __name__ == "__main__":
    df = compile_all_playoff_games(2001, 2025)
    print(df.head())
