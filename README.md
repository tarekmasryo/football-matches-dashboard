# ⚽ Football Matches 2024/2025 Dashboard

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)\
[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)\
[![Data License: CC BY-NC
4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-lightgrey.svg)](DATA_LICENSE)\
[![Made with ❤️ by Tarek
Masryo](https://img.shields.io/badge/Made%20by-Tarek%20Masryo-blue)](https://github.com/tarekmasryo)

------------------------------------------------------------------------

## 🎥 Live Preview

![Dashboard GIF](assets/football_dashboard.gif)

------------------------------------------------------------------------

## 📌 Overview

Interactive dashboard built with **Streamlit, Plotly, and Ag-Grid** to
explore the\
[Football Matches 2024/2025
Dataset](https://github.com/tarekmasryo/Football-Matches-Results-2024-2025-season/blob/main/data/football_matches_2024_2025.csv).

-   🏟️ 1,900+ matches from Premier League, La Liga, Serie A, Bundesliga,
    Ligue 1, and UEFA Champions League\
-   ⚽ KPIs: matches, goals, average goals per match\
-   📊 Standings auto-calculated (PTS → GD → GF)\
-   🔎 Filters by league, team, stage, match status, and date range\
-   🤝 Head-to-Head explorer for any two clubs

------------------------------------------------------------------------

## 📊 Dashboard Preview

### Overview (KPIs + Charts)

![Overview](assets/overview.png)

### Standings Table

![Standings](assets/standings.png)

### Team Explorer

![Teams](assets/team_explorer.png)

### Head-to-Head

![Head-to-Head](assets/head_to_head.png)

### Matches Table (Ag-Grid)

![Matches](assets/matches.png)

------------------------------------------------------------------------

## 🔑 Features

-   **Filters**: league, stage, match status, team, date range\
-   **KPIs**: matches, total goals, avg goals per match\
-   **Visuals**: bar charts, pie charts, timelines, histograms\
-   **Standings**: auto-ranked by Points, GD, GF\
-   **Team Explorer**: last 5 matches, attack & defense rankings\
-   **Head-to-Head**: compare two clubs (results, goals, timeline)\
-   **Ag-Grid Table**: sortable, filterable, downloadable

------------------------------------------------------------------------

## 🚀 Run Locally

Clone the repo and install requirements:

``` bash
git clone https://github.com/tarekmasryo/Football-Matches-Results-2024-2025-season.git
cd Football-Matches-Results-2024-2025-season
pip install -r requirements.txt
```

Run the app:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## ☁️ Deploy on Streamlit Cloud

You can deploy directly to [Streamlit
Cloud](https://streamlit.io/cloud).\
Make sure your **data source** points to the RAW CSV:

``` toml
# .streamlit/secrets.toml
DATA_URL = "https://raw.githubusercontent.com/tarekmasryo/Football-Matches-Results-2024-2025-season/main/data/football_matches_2024_2025.csv"
```

------------------------------------------------------------------------

## 📜 License & Attribution

-   **Code** → MIT License (see `LICENSE`)\
-   **Data** → Collected via
    [football-data.org](https://www.football-data.org/) API\
    Licensed under **CC BY-NC 4.0 (Attribution--NonCommercial)** ---
    Free for open research & educational use.\
    Commercial use is not permitted.

------------------------------------------------------------------------

## 🙌 Citation

If you use this dashboard or dataset, please credit as:

> Football Matches 2024/2025 Dashboard and Dataset by **Tarek Masryo**.\
> Code licensed under MIT. Data licensed under CC BY-NC 4.0.
