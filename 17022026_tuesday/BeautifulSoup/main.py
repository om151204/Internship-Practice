import requests
from bs4 import BeautifulSoup
import time


URL = "https://www.scrapethissite.com/pages/forms/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EducationalScraper/1.0"
}

MAX_DISPLAY = 12  # limit printed rows for demo


def scrape_nhl_teams():
    """
    Function to scrape the NHL teams page
    :return: teams: List of Data
    """
    print("NHL Teams Scraper\n")
    print(f"Fetching: {URL}\n")

    try:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        print(f"Success — Status: {response.status_code}")
        title = BeautifulSoup(response.text, "html.parser").title.string.strip()
        print(f"Page title: {title}\n")

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all team rows
        team_rows = soup.select("tr.team")

        if not team_rows:
            print("No rows with class='team' found. Inspect the page again.")
            return []

        print(f"Found {len(team_rows)} team rows.\n")

        teams = []

        for row in team_rows:
            # Extract each field safely
            name_cell = row.select_one("td.name")
            year_cell = row.select_one("td.year")
            wins_cell = row.select_one("td.wins")
            losses_cell = row.select_one("td.losses")
            otl_cell = row.select_one("td.ot-losses")
            pct_cell = row.select_one("td.pct")
            gf_cell = row.select_one("td.gf")
            ga_cell = row.select_one("td.ga")
            diff_cell = row.select_one("td.diff")

            # Skip if critical fields missing
            if not name_cell or not year_cell:
                continue

            team = {
                "Team": name_cell.get_text(strip=True),
                "Year": year_cell.get_text(strip=True),
                "W": wins_cell.get_text(strip=True) if wins_cell else "-",
                "L": losses_cell.get_text(strip=True) if losses_cell else "-",
                "OTL": otl_cell.get_text(strip=True) if otl_cell else "—",
                "Win%": pct_cell.get_text(strip=True) if pct_cell else "-",
                "GF": gf_cell.get_text(strip=True) if gf_cell else "-",
                "GA": ga_cell.get_text(strip=True) if ga_cell else "-",
                "+/-": diff_cell.get_text(strip=True) if diff_cell else "-"
            }

            teams.append(team)

        return teams

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return []
    except Exception as e:
        print(f"Parsing error: {e}")
        return []


def display_teams(teams):
    """
    Function to display teams
    :param teams: List of Data
    :return: None
    """
    if not teams:
        print("No teams extracted.")
        return

    print(f"{'#':<3} {'Team':<25} {'Year':<6} {'W':<4} {'L':<4} {'OTL':<5} {'Win%':<6} {'GF':<5} {'GA':<5} {'+/-':<5}")
    print("─" * 90)

    for i, t in enumerate(teams[:MAX_DISPLAY], 1):
        print(f"{i:<3} {t['Team']:<25} {t['Year']:<6} {t['W']:<4} {t['L']:<4} "
              f"{t['OTL']:<5} {t['Win%']:<6} {t['GF']:<5} {t['GA']:<5} {t['+/-']:<5}")

    if len(teams) > MAX_DISPLAY:
        print(f"\n... showing first {MAX_DISPLAY} of {len(teams)} teams")


# ────────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()

    teams_list = scrape_nhl_teams()
    display_teams(teams_list)

    print(f"\nCompleted in {time.time() - start_time:.2f} seconds\n")