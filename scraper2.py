import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SESSION = requests.Session()

def fetch_url(url, max_retries=5, backoff_factor=1.0):
    """
    GET with simple exponential backoff on 429 Too Many Requests.
    """
    for i in range(max_retries):
        resp = SESSION.get(url, headers=HEADERS)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            wait = backoff_factor * (2 ** i)
            print(f"429 from {url}, sleeping {wait:.1f}s (retry {i+1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise Exception(f"Failed to fetch {url} after {max_retries} retries (last status {resp.status_code})")

def get_seed_and_name(abbr: str, year: int, cache: dict) -> dict:
    key = f"{abbr}_{year}"
    if key in cache:
        return cache[key]
    url = f"https://www.basketball-reference.com/teams/{abbr}/{year}.html"
    resp = fetch_url(url)
    soup = BeautifulSoup(resp.content, "html.parser")
    # Team name
    h1 = soup.find("h1").get_text(strip=True)
    name = re.sub(r"^\d{4}\s+", "", h1)
    # Seed
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
    bracket_url = f"https://www.basketball-reference.com/playoffs/NBA_{year}.html"
    resp = fetch_url(bracket_url)
    soup = BeautifulSoup(resp.content, "html.parser")
    # Extract commented-out bracket HTML
    comments = "".join(str(c) for c in soup.find_all(string=lambda t: isinstance(t, Comment)))
    comment_soup = BeautifulSoup(comments, "html.parser")
    series_divs = soup.select("div.series_summary") + comment_soup.select("div.series_summary")
    if not series_divs:
        print(f"  ⚠️ No series summaries for {year}, skipping")
        return []
    # Unique team abbreviations
    playoff_abbrs = {
        a["href"].split("/")[2]
        for div in series_divs
        for a in div.find_all("a", href=True)
        if a["href"].startswith("/teams/")
    }
    games = []
    for abbr in playoff_abbrs:
        info = get_seed_and_name(abbr, year, cache)
        team1_name = info["name"]
        seed_home  = info["seed"]
        # Team game logs
        games_url = f"https://www.basketball-reference.com/teams/{abbr}/{year}_games.html"
        gr = fetch_url(games_url)
        gsoup = BeautifulSoup(gr.content, "html.parser")
        for c in gsoup.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()
        # Find playoffs table
        playoff_table = None
        for tbl in gsoup.find_all("table"):
            if tbl.get("id", "").lower().startswith("playoffs"):
                playoff_table = tbl
                break
        if playoff_table is None:
            continue
        # Parse rows
        for row in playoff_table.tbody.find_all("tr"):
            if row.get("class") and "thead" in row["class"]:
                continue
            # only home games
            loc = row.find("td", {"data-stat": "game_location"}).text.strip()
            if loc != "":
                continue
            opp_cell = row.find("td", {"data-stat": "opp_name"})
            if not opp_cell or not opp_cell.find("a"):
                continue
            opp_abbr = opp_cell.find("a")["href"].split("/")[2]
            opp_name = opp_cell.get_text(strip=True)
            # scores
            try:
                pts_h = int(row.find("td", {"data-stat": "pts"}).text)
                pts_a = int(row.find("td", {"data-stat": "opp_pts"}).text)
            except:
                continue
            opp_info = get_seed_and_name(opp_abbr, year, cache)
            seed_away = opp_info["seed"]
            games.append({
                "team1":      team1_name,
                "team2":      opp_name,
                "team1_id":   abbr,
                "team2_id":   opp_abbr,
                "team1_seed": seed_home,
                "team2_seed": seed_away,
                "team1_win":  pts_h > pts_a,
                "year":       year
            })
        time.sleep(0.3)
    return games

def compile_all_playoff_games(start_year=2001, end_year=2025) -> pd.DataFrame:
    cache = {}
    all_games = []
    for yr in range(start_year, end_year + 1):
        print(f"→ Scraping playoffs {yr}…")
        year_games = scrape_playoff_games_year(yr, cache)
        all_games.extend(year_games)
    df = pd.DataFrame(all_games)
    cols = ["team1","team2","team1_id","team2_id","team1_seed","team2_seed","team1_win","year"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv("game_by_game_2001_2025.csv", index=False)
    return df

if __name__ == "__main__":
    df = compile_all_playoff_games(2001, 2025)
    print(df.head())
