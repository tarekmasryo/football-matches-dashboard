# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==== Optional AgGrid (graceful fallback)
AGGRID_OK = True
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except Exception:
    AGGRID_OK = False

st.set_page_config(page_title="Football Matches 2024/2025 — Dashboard", layout="wide")

# ======================
# Constants / Data Source (fixed, remote)
# ======================
DATA_URL = (
    "https://raw.githubusercontent.com/tarekmasryo/Football-Matches-Results-2024-2025-season/"
    "main/data/football_matches_2024_2025.csv"
)

# ======================
# Utilities / URL State (compat new/old APIs)
# ======================
def get_qp():
    # Streamlit >=1.32
    try:
        return st.query_params
    except Exception:
        # Legacy fallback
        return st.experimental_get_query_params()

def set_qp(**kwargs):
    # Streamlit >=1.32 (mutable mapping)
    try:
        qp = st.query_params
        for k, v in kwargs.items():
            if v is None:
                try:
                    qp.pop(k, None)
                except Exception:
                    pass
            else:
                qp[k] = v
    except Exception:
        # Legacy API requires full dict
        cur = st.experimental_get_query_params()
        for k, v in list(kwargs.items()):
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        st.experimental_set_query_params(**cur)

# =========
# Load Data (with timeout/retries + dtype normalization)
# =========
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(path_or_buffer):
    import io, time

    def _postprocess_df(df):
        for c in ["fulltime_home", "fulltime_away", "total_goals", "home_points", "away_points"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        if "date_utc" in df.columns:
            df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")

        if {"fulltime_home", "fulltime_away"}.issubset(df.columns):
            df["goal_difference"] = df["fulltime_home"] - df["fulltime_away"]
        else:
            df["goal_difference"] = np.nan

        return df

    def _read_csv_from_bytes(b):
        df = pd.read_csv(io.BytesIO(b))
        return _postprocess_df(df)

    # Uploaded file-like object
    if not isinstance(path_or_buffer, str):
        return _postprocess_df(pd.read_csv(path_or_buffer))

    # Remote URL with timeout & retries
    url = path_or_buffer
    tries, last_err = 3, None
    for _ in range(tries):
        try:
            # Prefer requests if available (for headers/timeout)
            try:
                import requests
                resp = requests.get(url, timeout=10, headers={"User-Agent": "streamlit-app"})
                resp.raise_for_status()
                return _read_csv_from_bytes(resp.content)
            except Exception:
                # Fallback: pandas direct read
                df = pd.read_csv(url)
                return _postprocess_df(df)
        except Exception as e:
            last_err = e
            time.sleep(1.5)

    st.error(f"Failed to fetch dataset from URL after {tries} tries.\n{last_err}")
    st.stop()

# ==================
# Sidebar — Data Source (Upload or Default)
# ==================
st.sidebar.header("📥 Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV (same schema)", type="csv")

with st.sidebar:
    status_placeholder = st.empty()
    status_placeholder.info("Fetching data..." if uploaded_file is None else "Reading uploaded data...")

df = load_data(uploaded_file if uploaded_file is not None else DATA_URL)
status_placeholder.success("Data loaded")

# ==================
# Sidebar — Filters
# ==================
st.sidebar.header("⚙️ Filters")

# League filter
leagues = sorted(df["competition_name"].dropna().unique().tolist()) if "competition_name" in df.columns else []
default_leagues = leagues.copy()
qp = get_qp()
if "lg" in qp and leagues:
    qs_val = qp.get("lg")
    # qp may return list in legacy API
    if isinstance(qs_val, list):
        qs_val = qs_val[0] if qs_val else ""
    qs_leagues = [x for x in str(qs_val).split("|") if x in leagues]
    if qs_leagues:
        default_leagues = qs_leagues
selected_leagues = st.sidebar.multiselect("League(s)", leagues, default=default_leagues)

# Stage / Status
stages_all = sorted(df["stage"].dropna().unique().tolist()) if "stage" in df.columns else []
selected_stages = st.sidebar.multiselect(
    "Stage", stages_all, default=(stages_all if len(stages_all) <= 6 else [])
)

statuses_all = sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else []
default_status = ["FINISHED"] if "FINISHED" in statuses_all else statuses_all
selected_status = st.sidebar.multiselect("Status", statuses_all, default=default_status)

# Teams
if {"home_team", "away_team"}.issubset(df.columns):
    teams_all = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]])).tolist())
else:
    teams_all = []
selected_teams = st.sidebar.multiselect("Team(s)", teams_all)

# Date range (UTC only)
min_date = pd.to_datetime(df["date_utc"].min()) if "date_utc" in df.columns else None
max_date = pd.to_datetime(df["date_utc"].max()) if "date_utc" in df.columns else None
if min_date is not None and max_date is not None:
    dr_default = (min_date.date(), max_date.date())
    ds_raw, de_raw = qp.get("ds"), qp.get("de")
    if isinstance(ds_raw, list):
        ds_raw = ds_raw[0] if ds_raw else None
    if isinstance(de_raw, list):
        de_raw = de_raw[0] if de_raw else None
    if ds_raw and de_raw:
        try:
            dr_default = (pd.to_datetime(ds_raw).date(), pd.to_datetime(de_raw).date())
        except Exception:
            pass
    dr = st.sidebar.date_input("Date range (UTC)", dr_default)
    if isinstance(dr, tuple) and len(dr) == 2:
        start_date, end_date = pd.to_datetime(dr[0]), pd.to_datetime(dr[1]) + pd.Timedelta(days=1)
    else:
        start_date, end_date = min_date, max_date + pd.Timedelta(days=1)
else:
    start_date, end_date = None, None
    dr = None

# Apply filters (global)
df_f = df.copy()
if selected_leagues and "competition_name" in df_f.columns:
    df_f = df_f[df_f["competition_name"].isin(selected_leagues)]
if selected_stages and "stage" in df_f.columns:
    df_f = df_f[df_f["stage"].isin(selected_stages)]
if selected_status and "status" in df_f.columns:
    df_f = df_f[df_f["status"].isin(selected_status)]
if selected_teams and {"home_team", "away_team"}.issubset(df_f.columns):
    df_f = df_f[(df_f["home_team"].isin(selected_teams)) | (df_f["away_team"].isin(selected_teams))]
if start_date is not None and end_date is not None and "date_utc" in df_f.columns:
    # date_utc already tz-aware -> drop tz for comparison
    df_f = df_f[
        (df_f["date_utc"].dt.tz_localize(None) >= start_date)
        & (df_f["date_utc"].dt.tz_localize(None) < end_date)
    ]

# Default sort by latest date first
if "date_utc" in df_f.columns:
    df_f = df_f.sort_values("date_utc", ascending=False)

# Save state to URL (shareable view)
set_qp(
    lg="|".join(selected_leagues) if selected_leagues else None,
    ds=str(dr[0]) if (min_date is not None and max_date is not None and isinstance(dr, tuple) and len(dr) == 2) else None,
    de=str(dr[1]) if (min_date is not None and max_date is not None and isinstance(dr, tuple) and len(dr) == 2) else None,
)

# Empty state guard
if df_f.empty:
    st.warning("No matches for current filters. Try expanding leagues/teams/date range.")
    st.stop()

# ====================
# Helper Calculations
# ====================
def add_team_perspective(df_):
    cols = [
        "competition_name","matchday","date_utc","home_team","away_team",
        "fulltime_home","fulltime_away","match_outcome","home_points","away_points",
    ]
    home = df_[cols].copy()

    home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "fulltime_home": "gf",
            "fulltime_away": "ga",
            "home_points": "points",
        },
        inplace=True,
    )
    home["venue"] = "Home"
    for c in ["gf", "ga", "points"]:
        home[c] = pd.to_numeric(home[c], errors="coerce").fillna(0)
    home["result"] = np.where(home["gf"] > home["ga"], "W", np.where(home["gf"] < home["ga"], "L", "D"))

    away = home.copy()
    away[["team", "opponent", "gf", "ga", "points"]] = df_[
        ["away_team", "home_team", "fulltime_away", "fulltime_home", "away_points"]
    ].to_numpy()
    for c in ["gf", "ga", "points"]:
        away[c] = pd.to_numeric(away[c], errors="coerce").fillna(0)
    away["venue"] = "Away"
    away["result"] = np.where(away["gf"] > away["ga"], "W", np.where(away["gf"] < away["ga"], "L", "D"))

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["gd"] = long_df["gf"] - long_df["ga"]
    return long_df

def compute_standings(df_):
    long_df = add_team_perspective(df_)
    g = long_df.groupby(["competition_name", "team"], as_index=False).agg(
        MP=("team", "count"),
        W=("result", lambda x: (x == "W").sum()),
        D=("result", lambda x: (x == "D").sum()),
        L=("result", lambda x: (x == "L").sum()),
        GF=("gf", "sum"),
        GA=("ga", "sum"),
        GD=("gd", "sum"),
        PTS=("points", "sum"),
    )
    for c in ["MP", "W", "D", "L", "GF", "GA", "GD", "PTS"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)
    g = g.sort_values(["competition_name", "PTS", "GD", "GF"], ascending=[True, False, False, False])
    g["Rank"] = g.groupby("competition_name").cumcount() + 1
    return g[["competition_name", "Rank", "team", "MP", "W", "D", "L", "GF", "GA", "GD", "PTS"]]

def team_form(df_, team, n=5):
    long_df = add_team_perspective(df_)
    t = long_df[long_df["team"] == team].sort_values("date_utc")
    last = t.tail(n)
    return last[["date_utc", "opponent", "venue", "gf", "ga", "result", "points"]]

def head_to_head_matrix(df_):
    long_df = add_team_perspective(df_)
    wins = long_df.copy()
    wins["win"] = (wins["result"] == "W").astype(int)
    mat = wins.pivot_table(index="team", columns="opponent", values="win", aggfunc="sum", fill_value=0)
    return mat

# =========
# KPIs Row
# =========
st.title("⚽ Football Matches 2024/2025 — Dashboard")

total_matches = len(df_f)
if "total_goals" in df_f.columns:
    total_goals = pd.to_numeric(df_f["total_goals"], errors="coerce").fillna(0).sum()
else:
    total_goals = (
        pd.to_numeric(df_f.get("fulltime_home", 0), errors="coerce").fillna(0).sum()
        + pd.to_numeric(df_f.get("fulltime_away", 0), errors="coerce").fillna(0).sum()
    )
total_goals = int(total_goals)
avg_goals = round(total_goals / total_matches, 2) if total_matches else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Matches", total_matches)
c2.metric("Goals", total_goals)
c3.metric("Avg Goals/Match", avg_goals)

# =======
# TABS UI
# =======
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Overview",
    "🏆 Standings",
    "🧭 Team Explorer",
    "🤝 Head-to-Head",
    "📋 Matches",
])

# -------------
# Tab 1: Overview
# -------------
with tab1:
    colA, colB = st.columns([1.2, 1])
    with colA:
        if "competition_name" in df_f.columns and "total_goals" in df_f.columns:
            gb_ = df_f.groupby("competition_name", as_index=False)["total_goals"].sum()
            fig1 = px.bar(gb_, x="competition_name", y="total_goals", title="Goals by League", text="total_goals")
            fig1.update_traces(textposition="outside")
            st.plotly_chart(fig1, use_container_width=True)
    with colB:
        if "match_outcome" in df_f.columns and "status" in df_f.columns:
            pie_df = (
                df_f[df_f["status"] == "FINISHED"]["match_outcome"]
                .value_counts()
                .rename_axis("result")
                .reset_index(name="count")
            )
            if not pie_df.empty:
                fig2 = px.pie(pie_df, names="result", values="count", title="Result Distribution (Finished)")
                st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns([1, 1])
    with colC:
        if "date_utc" in df_f.columns and "total_goals" in df_f.columns:
            df_t = df_f.copy()
            df_t["date_only"] = df_t["date_utc"].dt.date
            tl = df_t.groupby("date_only", as_index=False)["total_goals"].sum()
            fig3 = px.line(tl, x="date_only", y="total_goals", title="Goals Timeline (by Date, UTC)")
            st.plotly_chart(fig3, use_container_width=True)
    with colD:
        if {"matchday", "competition_name", "total_goals"}.issubset(df_f.columns):
            md = (
                df_f.groupby(["competition_name", "matchday"]).agg(
                    Goals=("total_goals", "sum"), Matches=("competition_name", "count")
                )
            ).reset_index()
            md["AvgGoalsPerMatch"] = md["Goals"] / md["Matches"]
            fig4 = px.line(
                md,
                x="matchday",
                y="AvgGoalsPerMatch",
                color="competition_name",
                markers=True,
                title="Average Goals per Matchday",
            )
            st.plotly_chart(fig4, use_container_width=True)

    if "total_goals" in df_f.columns:
        st.subheader("Goals per Match — Distribution")
        fig5 = px.histogram(df_f, x="total_goals", nbins=10, marginal="rug", title="Histogram: Total Goals per Match")
        st.plotly_chart(fig5, use_container_width=True)

# ---------------
# Tab 2: Standings
# ---------------
with tab2:
    st.caption("Ranked by Points ➝ Goal Difference ➝ Goals For")
    league_for_table = st.selectbox("League table for:", ["All"] + leagues, index=0)
    table = compute_standings(df_f)
    if league_for_table != "All":
        table = table[table["competition_name"] == league_for_table]

    sort_choice = st.radio("Sort by", ["PTS", "GD", "GF"], horizontal=True, index=0)
    asc_map = {"PTS": False, "GD": False, "GF": False}
    table = table.sort_values(
        ["competition_name", sort_choice, "GD", "GF"],
        ascending=[True, asc_map[sort_choice], False, False],
    )
    table["Rank"] = table.groupby("competition_name").cumcount() + 1

    by_league = st.toggle("Split by League", value=(league_for_table == "All"))
    if by_league and league_for_table == "All":
        for lg in table["competition_name"].unique():
            st.markdown(f"#### {lg}")
            t = table[table["competition_name"] == lg].drop(columns=["competition_name"])
            st.dataframe(t, use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            table if league_for_table == "All" else table.drop(columns=["competition_name"]),
            use_container_width=True,
            hide_index=True,
        )

    csv_standings = table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Standings CSV", csv_standings, "standings.csv", "text/csv")

# -------------------
# Tab 3: Team Explorer
# -------------------
with tab3:
    sel_team = st.selectbox("Pick a Team", options=teams_all if teams_all else [])
    if sel_team:
        st.markdown(f"### {sel_team} — Last 5 matches")
        form_df = team_form(df_f, sel_team, n=5)

        def color_result(val):
            color = {"W": "#16a34a", "D": "#9ca3af", "L": "#ef4444"}.get(val, "#e5e7eb")
            return f"color: {color}; font-weight: 700;"

        st.dataframe(
            form_df.rename(columns={"date_utc": "date"})
            .style.applymap(lambda v: "font-weight:700;" if v in ["W", "D", "L"] else "")
            .applymap(color_result, subset=["result"]),
            use_container_width=True,
            hide_index=True,
        )

        long_df = add_team_perspective(df_f)

        agg = long_df.groupby("team", as_index=False).agg(
            MP=("team", "count"),
            GF=("gf", "sum"),
            GA=("ga", "sum"),
        )

        for c in ["MP", "GF", "GA"]:
            agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0)

        mp_safe = agg["MP"].replace(0, np.nan)
        agg["GF/Match"] = (agg["GF"] / mp_safe).round(2).fillna(0)
        agg["GA/Match"] = (agg["GA"] / mp_safe).round(2).fillna(0)

        cL, cR = st.columns(2)
        with cL:
            top_attack = agg.sort_values("GF/Match", ascending=False).head(10)
            figA = px.bar(top_attack, x="team", y="GF/Match", title="Top 10 Attacks (GF/Match)")
            st.plotly_chart(figA, use_container_width=True)
        with cR:
            top_def = agg.sort_values("GA/Match").head(10)
            figB = px.bar(top_def, x="team", y="GA/Match", title="Best 10 Defenses (GA/Match, lower better)")
            st.plotly_chart(figB, use_container_width=True)

# -------------------
# Tab 4: Head-to-Head
# -------------------
with tab4:
    st.caption("Compare two teams directly: results, goals and recent form.")
    t1 = st.selectbox("Team A", teams_all, index=0)
    t2 = st.selectbox("Team B", teams_all, index=min(1, len(teams_all) - 1))
    if t1 == t2:
        st.info("Pick two different teams to compare.")
    else:
        h2h = df_f[
            ((df_f["home_team"] == t1) & (df_f["away_team"] == t2))
            | ((df_f["home_team"] == t2) & (df_f["away_team"] == t1))
        ].copy()
        if h2h.empty:
            st.warning("No head-to-head matches under current filters.")
        else:
            def count_results(df_, a, b):
                ah = df_["home_team"] == a
                aa = df_["away_team"] == a
                a_wins = int(((ah) & (df_["fulltime_home"] > df_["fulltime_away"])).sum() +
                             ((aa) & (df_["fulltime_away"] > df_["fulltime_home"])).sum())
                b_wins = int(((ah) & (df_["fulltime_home"] < df_["fulltime_away"])).sum() +
                             ((aa) & (df_["fulltime_away"] < df_["fulltime_home"])).sum())
                draws  = int((df_["fulltime_home"] == df_["fulltime_away"]).sum())
                ga = int(((ah) * df_["fulltime_home"] + (aa) * df_["fulltime_away"]).sum())
                gb = int(((ah) * df_["fulltime_away"] + (aa) * df_["fulltime_home"]).sum())
                return a_wins, draws, b_wins, ga, gb

            aW, D, bW, GA, GB = count_results(h2h, t1, t2)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{t1} Wins", aW)
            c2.metric("Draws", D)
            c3.metric(f"{t2} Wins", bW)
            c1.metric(f"{t1} Goals", GA)
            c3.metric(f"{t2} Goals", GB)

            if "date_utc" in h2h.columns:
                figH = px.scatter(
                    h2h.sort_values("date_utc"),
                    x="date_utc",
                    y="total_goals",
                    hover_data=[
                        "competition_name",
                        "home_team",
                        "fulltime_home",
                        "fulltime_away",
                        "away_team",
                    ],
                    title=f"Head-to-Head Goals Over Time — {t1} vs {t2} (UTC)",
                )
                st.plotly_chart(figH, use_container_width=True)

            st.subheader("Head-to-Head Matches")
            cols = [
                "date_utc","competition_name","home_team","fulltime_home","fulltime_away",
                "away_team","matchday","stage","status",
            ]
            show_cols = [c for c in cols if c in h2h.columns]
            st.dataframe(h2h[show_cols].sort_values("date_utc"), use_container_width=True, hide_index=True)

# --------------
# Tab 5: Matches (Ag-Grid + in-tab filters & better UX)
# --------------
with tab5:
    st.caption("Filtered matches below. Use sidebar to slice.")

    cols_order = [
        "date_utc","competition_name","matchday","home_team",
        "fulltime_home","fulltime_away","away_team",
        "match_outcome","stage","status","total_goals",
    ]
    show_cols = [c for c in cols_order if c in df_f.columns] + [c for c in df_f.columns if c not in cols_order]
    df_matches = df_f[show_cols].copy()

    if "date_utc" in df_matches.columns:
        df_matches.insert(0, "date_str_utc", df_matches["date_utc"].dt.strftime("%Y-%m-%d %H:%M"))

    # In-tab refiners
    comp_opts = sorted(df_f["competition_name"].dropna().unique().tolist()) if "competition_name" in df_f.columns else []
    comp_sel = st.multiselect("Competition(s) (refine this table)", comp_opts, default=comp_opts)

    status_opts = sorted(df_f["status"].dropna().unique().tolist()) if "status" in df_f.columns else []
    status_sel = st.multiselect("Status (refine)", status_opts, default=status_opts)

    outcome_opts = sorted(df_f["match_outcome"].dropna().unique().tolist()) if "match_outcome" in df_f.columns else []
    outcome_sel = st.multiselect("Result/Outcome (refine)", outcome_opts, default=outcome_opts)

    q = st.text_input("Quick filter (search in table)", "")

    if comp_sel and "competition_name" in df_matches.columns:
        df_matches = df_matches[df_matches["competition_name"].isin(comp_sel)]
    if status_sel and "status" in df_matches.columns:
        df_matches = df_matches[df_matches["status"].isin(status_sel)]
    if outcome_sel and "match_outcome" in df_matches.columns:
        df_matches = df_matches[df_matches["match_outcome"].isin(outcome_sel)]

    if AGGRID_OK:
        gb = GridOptionsBuilder.from_dataframe(df_matches)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(
            editable=False, groupable=True, filter=True, sortable=True, resizable=True, floatingFilter=True,
        )
        if "date_utc" in df_matches.columns:
            gb.configure_column("date_utc", hide=True)
        gb.configure_grid_options(
            rememberGroupStateWhenNewData=True,
            suppressRowClickSelection=True,
            rowSelection="multiple",
            animateRows=True,
            quickFilterText=q,
        )
        gridOptions = gb.build()

        # allow_unsafe_jscode=True prevents JsCode serialization issues (even if not used)
        grid_resp = AgGrid(
            df_matches,
            gridOptions=gridOptions,
            theme="streamlit",
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.NO_UPDATE,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
        )

        df_visible = pd.DataFrame(grid_resp["data"]) if isinstance(grid_resp, dict) and "data" in grid_resp else df_matches
        csv_matches = df_visible.to_csv(index=False).encode("utf-8")
    else:
        st.info("AgGrid not available on this environment — falling back to Streamlit table.")
        st.dataframe(df_matches, use_container_width=True, hide_index=True)
        csv_matches = df_matches.to_csv(index=False).encode("utf-8")

    st.download_button("⬇️ Download Filtered Matches CSV", csv_matches, "filtered_matches.csv", "text/csv")
