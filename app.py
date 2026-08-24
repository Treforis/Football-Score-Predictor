import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Football Predictor", layout="wide")

league = st.sidebar.selectbox("League",["PD","PL","BL1","SA","FL1"], format_func=lambda x: {"PD": "La Liga", "PL": "Premiere League", "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1"}[x])


fixtures = pd.read_csv(f"artifacts/{league}/fixtures.csv")
fixtures['kick'] = (pd.to_datetime(fixtures['utc'], utc=True)
                      .dt.tz_convert('America/Toronto')
                      .dt.tz_localize(None))
fixtures['dt'] = fixtures['kick'].dt.normalize()
maxmd = int(fixtures['matchday'].max())
weekly = pd.read_csv(f"artifacts/{league}/weekly.csv")
st.caption(f"Last updated: {open('artifacts/last_update.txt').read()[:16]}")

names = {"PD": "La Liga", "PL": "Premiere League", "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1"}
st.title(f"{names[league]} 2026-27 Predictor")
tab1, tab2, tab3 = st.tabs(["Results","Matchdays", "Table"])

def badge(url, width=34):
    if isinstance(url,str) and url.startswith('http'):
        return f"<img  src='{url}' width = '{width}' style= 'vertical-align:middle'>"
    return ""

def day_picker(d, key):
    days = sorted(d['dt'].dt.date.unique())
    if len(days) == 0:
        return d, None

    i = st.session_state.get(key, 0)
    i = max(0, min(i, len(days) - 1))

    c1, c2, c3 = st.columns([1, 4, 1])

    if c1.button("‹ Prev", key=f"{key}_p", disabled=(i == 0)):
        i -= 1
    if c3.button("Next ›", key=f"{key}_n", disabled=(i == len(days) - 1)):
        i += 1

    i = max(0, min(i, len(days) - 1))
    st.session_state[key] = i

    day = days[i]
    c2.markdown(
        f"<div style='text-align:center;font-size:1.2em'>"
        f"<b>{day.strftime('%a %d %b %Y')}</b><br>"
        f"<span style='font-size:0.7em;opacity:0.6'>{i + 1} of {len(days)}</span></div>",
        unsafe_allow_html=True)

    return d[d['dt'].dt.date == day], day

def match_card(f,middle):
    c1,c2,c3 = st.columns([4,3,4])
    c1.markdown(f"<div style='text-align:right'> {badge(f.home_crest)}"
                f"<b>{f.home}</b></div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='text-align:center'>{middle}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div><b>{f.away}</b> {badge(f.away_crest)}</div>", unsafe_allow_html=True)

with tab1:
    today = pd.Timestamp.today().normalize()

    finished = fixtures[fixtures['dt'] < today]
    live     = fixtures[fixtures['dt'] == today]
    upcoming = fixtures[fixtures['dt'] > today]

    t_fin, t_live, t_up = st.tabs([
        f"Finished ({len(finished)})",
        f"Live ({len(live)})",
        f"Upcoming ({len(upcoming)})",
    ])

    with t_fin:
        d, _ = day_picker(finished.sort_values('dt', ascending=False), "d_fin")
        for f in d.itertuples():
            st.caption(f"MD {f.matchday} - {f.kick.strftime('%d %b %Y')}")
            if pd.notna(f.home_goals):
                mid = f"<span style= 'font-sizeL1.6em'><b>{int(f.home_goals)} - {int(f.away_goals)}</b></span>"
            else:
                mid = "<span style='opacity:0.5'>no result</span>"
            match_card(f,mid)
            st.divider()

    with t_live:
        if len(live) == 0:
            st.info("No matches today.")
        for f in live.itertuples():
            if pd.notna(f.home_goals):
                mid = f"<span style='font-size:1.6em'><b>{int(f.home_goals)} – {int(f.away_goals)}</b></span>"
            else:
                mid = (f"<span style='font-size:1.4em'>{f.kick.strftime('%H:%M')}</span><br>"
                       f"<span style='font-size:0.8em'>H {f.p_H:.0%} · D {f.p_D:.0%} · A {f.p_A:.0%}</span>")
            match_card(f, mid)
            st.divider()

    with t_up:
        d, _ = day_picker(upcoming.sort_values('dt'), "d_up")
        n = st.number_input("Show next", 5, 100, 20, 5)
        for f in d.itertuples():
            st.caption(f"MD {f.matchday} · {f.kick.strftime('%a %d %b, %H:%M')}")
            mid = (f"<span style='font-size:1.4em'>{round(f.xg_home)} – {round(f.xg_away)}</span><br>"
                   f"<span style='font-size:0.8em'>H {f.p_H:.0%} · D {f.p_D:.0%} · A {f.p_A:.0%}</span>")
            match_card(f, mid)
            st.divider()

with tab2:
    md = st.number_input("Matchday", min_value=1, max_value=maxmd, value=1, step=1)
    week = fixtures[fixtures['matchday'] == md]

    st.caption(f"{len(week)} fixtures · {week['date'].min()} to {week['date'].max()}")

    for f in week.itertuples():
        c1, c2, c3 = st.columns([3, 3, 3])

        c1.markdown(f"**{f.home}**")
        c3.markdown(f"**{f.away}**")

        if f.status == 'FINISHED':
            c2.markdown(f"### {int(f.home_goals)} – {int(f.away_goals)}")
        else:
            c2.markdown(
                f"### {round(f.xg_home)} – {round(f.xg_away)}\n"
                f"_{f.xg_home} – {f.xg_away}_  \n"
                f"H {f.p_H:.0%} · D {f.p_D:.0%} · A {f.p_A:.0%}"
            )
        st.divider()

with tab3:
    upto = st.slider("Through matchday", 1, maxmd, maxmd)
    w = weekly[weekly['matchday'] <= upto].copy()
    done = w['played'] == 1

    w['W']    = np.where(done, (w['pts'] == 3).astype(int), w['xW'])
    w['D']    = np.where(done, (w['pts'] == 1).astype(int), w['xD'])
    w['L']    = np.where(done, (w['pts'] == 0).astype(int), w['xL'])
    w['proj'] = np.where(done, w['pts'], w['exp_pts'])

    t = w.groupby('team').agg(
        P=('matchday', 'count'),
        Played=('played', 'sum'),
        W=('W', 'sum'),
        D=('D', 'sum'),
        L=('L', 'sum'),
        Proj=('proj', 'sum'),
        Pts=('pts', 'sum'),
    )

    t['Proj'] = t['Proj'].round().astype(int)
    t[['W', 'D', 'L']] = t[['W', 'D', 'L']].round(1)
    t['Played'] = t['Played'].astype(int)
    t['Pts'] = t['Pts'].astype(int)

    t = t.sort_values('Proj', ascending=False)
    t.index.name = 'Team'
    t.insert(0, 'Pos', range(1, len(t) + 1))

    st.dataframe(t, use_container_width=True)
    