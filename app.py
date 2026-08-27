"""
app.py — the Streamlit front end.

Read-only: it loads two CSVs per league and displays them, never touching
the models or the API. That's why the deployed app needs no token, no
scikit-learn and no training data.

Streamlit reruns this whole file on every interaction, so keeping it to
two CSV reads is what makes that fast enough.
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="Football Predictor", layout="wide")

# Streamlit runs server-side, so it can't know the visitor's timezone. This
# asks the browser. Returns None on the first run before the browser answers,
# hence the fallback; Streamlit reruns once the real value arrives.
TZ = streamlit_js_eval(js_expression="Intl.DateTimeFormat().resolvedOptions().timezone", key="tz") or "America/Toronto"

league = st.sidebar.selectbox("League",["PD","PL","BL1","SA","FL1"], format_func=lambda x: {"PD": "La Liga", "PL": "Premiere League", "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1"}[x])


fixtures = pd.read_csv(f"artifacts/{league}/fixtures.csv")
# API gives UTC. Convert, then drop the timezone so these stay comparable
# with the naive timestamps below (mixing aware and naive raises).
fixtures['kick'] = (pd.to_datetime(fixtures['utc'], utc=True)
                      .dt.tz_convert(TZ)
                      .dt.tz_localize(None))
fixtures['dt'] = fixtures['kick'].dt.normalize()
# Derived, not hardcoded: Bundesliga and Ligue 1 have 34 matchdays, not 38.
maxmd = int(fixtures['matchday'].max())
weekly = pd.read_csv(f"artifacts/{league}/weekly.csv")
st.caption(f"Last updated: {open('artifacts/last_update.txt').read()[:16]}")

names = {"PD": "La Liga", "PL": "Premiere League", "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1"}
st.title(f"{names[league]} 2026-27 Predictor")
tab1, tab2, tab3 = st.tabs(["Results","Matchdays", "Table"])

def badge(url, width=34):
    """An <img> tag for a club crest, or empty string if there isn't one.

    Raw HTML rather than st.image because some crests are SVG. The guard
    matters: DEFAULT teams have no crest, and a missing value would print
    "nan" into the page.
    """
    if isinstance(url,str) and url.startswith('http'):
        return f"<img  src='{url}' width = '{width}' style= 'vertical-align:middle'>"
    return ""

def day_picker(d, key):
    """Prev/Next buttons; returns only the fixtures on the selected day.

    Steps through days that have fixtures, not calendar days, so empty
    midweeks are skipped. The clicks are handled in on_click callbacks
    because those fire before the rerun, so the buttons' disabled state
    matches what's on screen. Reading the return value leaves them a step
    behind.
    """
    days = sorted(d['dt'].dt.date.unique())
    if len(days) == 0:
        return d, None

    i = st.session_state.get(key, 0)
    # Clamp before the click too, in case another league left a stale index.
    i = max(0, min(i, len(days) - 1))

    # pop, not get: consume the flag so one click doesn't repeat next rerun.
    if st.session_state.pop(f"{key}_go_p", False):
        i -= 1
    if st.session_state.pop(f"{key}_go_n", False):
        i += 1

    i = max(0, min(i, len(days) - 1))
    st.session_state[key] = i

    c1, c2, c3 = st.columns([1, 4, 1])
    c1.button("‹ Prev", key=f"{key}_p", disabled=(i == 0),
              on_click=lambda: st.session_state.update({f"{key}_go_p": True}))
    c3.button("Next ›", key=f"{key}_n", disabled=(i == len(days) - 1),
              on_click=lambda: st.session_state.update({f"{key}_go_n": True}))

    day = days[i]
    c2.markdown(
        f"<div style='text-align:center;font-size:1.2em'>"
        f"<b>{day.strftime('%a %d %b %Y')}</b><br>"
        f"<span style='font-size:0.7em;opacity:0.6'>{i + 1} of {len(days)}</span></div>",
        unsafe_allow_html=True)

    return d[d['dt'].dt.date == day], day

def match_card(f, middle):
    """One fixture as a head-to-head row.

    Built as a single flexbox block rather than st.columns, because
    Streamlit stacks columns vertically below ~640px and the card falls
    apart into three separate rows on a phone. Flex stays horizontal at
    any width.
    """
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;width:100%'>"
        f"  <div style='flex:1;text-align:right;display:flex;align-items:center;"
        f"              justify-content:flex-end;gap:6px;min-width:0'>"
        f"    <b style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{f.home}</b>"
        f"    {badge(f.home_crest, 26)}"
        f"  </div>"
        f"  <div style='flex:0 0 auto;text-align:center;line-height:1.25'>{middle}</div>"
        f"  <div style='flex:1;text-align:left;display:flex;align-items:center;"
        f"              gap:6px;min-width:0'>"
        f"    {badge(f.away_crest, 26)}"
        f"    <b style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{f.away}</b>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True)

with tab1:
    today = pd.Timestamp.now(tz='America/Toronto').normalize().tz_localize(None)

    # Split on date, not status: the API's status lags, and matches played
    # days ago have been seen still marked TIMED. Using <, == and > against
    # one value means every fixture lands in exactly one bucket.
    finished = fixtures[fixtures['dt'] < today]
    live     = fixtures[fixtures['dt'] == today]
    upcoming = fixtures[fixtures['dt'] > today]

    t_fin, t_live, t_up = st.tabs([
        f"Finished ({len(finished)})",
        f"Live ({len(live)})",
        f"Upcoming ({len(upcoming)})",
    ])

    with t_fin:
        # Newest first: recent results are what you actually want to see.
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
            # Check goals, not status: a result can land before status catches up.
            if pd.notna(f.home_goals):
                mid = f"<span style='font-size:1.6em'><b>{int(f.home_goals)} – {int(f.away_goals)}</b></span>"
            else:
                # Not started, so show kickoff time. A predicted scoreline for
                # a match already under way is noise.
                mid = (f"<span style='font-size:1.4em'>{f.kick.strftime('%H:%M')}</span><br>"
                       f"<span style='font-size:0.8em'>H {f.p_H:.0%} · D {f.p_D:.0%} · A {f.p_A:.0%}</span>")
            match_card(f, mid)
            st.divider()

    with t_up:
        # Oldest first, so you land on the next day coming up.
        d, _ = day_picker(upcoming.sort_values('dt'), "d_up")
        for f in d.itertuples():
            st.caption(f"MD {f.matchday} · {f.kick.strftime('%a %d %b, %H:%M')}")
            # The probabilities carry more than the rounded scoreline:
            # 1.6-1.4 and 2.4-0.6 both display as 2-1.
            mid = (f"<span style='font-size:1.4em'>{round(f.xg_home)} – {round(f.xg_away)}</span><br>"
                   f"<span style='font-size:0.8em'>H {f.p_H:.0%} · D {f.p_D:.0%} · A {f.p_A:.0%}</span>")
            match_card(f, mid)
            st.divider()

with tab2:
    md = st.number_input("Matchday", min_value=1, max_value=maxmd, value=1, step=1)
    week = fixtures[fixtures['matchday'] == md]

    st.caption(f"{len(week)} fixtures · {week['dt'].min().strftime('%d %b')} "f"to {week['dt'].max().strftime('%d %b')}")

    for f in week.itertuples():
        st.caption(f.kick.strftime('%a %d %b, %H:%M'))

        # The one view showing the real score and the prediction together,
        # which makes it the most useful for judging the model.
        if pd.notna(f.home_goals):
            mid = (f"<span style='font-size:1.8em'><b>{int(f.home_goals)} – {int(f.away_goals)}</b></span><br>"
                   f"<span style='font-size:0.75em;opacity:0.55'>predicted {round(f.xg_home)} – {round(f.xg_away)}</span>")
        else:
            mid = (f"<span style='font-size:1.5em'>{round(f.xg_home)} – {round(f.xg_away)}</span><br>"
                   f"<span style='font-size:0.8em'>H {f.p_H:.0%} · D {f.p_D:.0%} · A {f.p_A:.0%}</span>")
        match_card(f, mid)
        st.divider()

with tab3:
    upto = st.slider("Through matchday", 1, maxmd, maxmd)
    w = weekly[weekly['matchday'] <= upto].copy()
    done = w['played'] == 1

    # Blend actual and expected: played matches contribute their real result,
    # unplayed ones their probabilities. Without this a team that won 3-0
    # shows zero wins, because its pre-match probabilities round to nothing.
    w['W']    = np.where(done, (w['pts'] == 3).astype(int), w['xW'])
    w['D']    = np.where(done, (w['pts'] == 1).astype(int), w['xD'])
    w['L']    = np.where(done, (w['pts'] == 0).astype(int), w['xL'])
    w['proj'] = np.where(done, w['pts'], w['exp_pts'])

    # weekly.csv is one row per team per match, so grouping by team gives a table.
    t = w.groupby('team').agg(
        P=('matchday', 'count'),
        Played=('played', 'sum'),
        W=('W', 'sum'),
        D=('D', 'sum'),
        L=('L', 'sum'),
        Proj=('proj', 'sum'),
        Pts=('pts', 'sum'),
    )

    # W/D/L keep a decimal because they're part expectation; rounding them to
    # whole numbers early in the season collapses everything to zero.
    t['Proj'] = t['Proj'].round().astype(int)
    t[['W', 'D', 'L']] = t[['W', 'D', 'L']].round(1)
    t['Played'] = t['Played'].astype(int)
    t['Pts'] = t['Pts'].astype(int)

    # Sort before numbering, or Pos won't match the displayed order.
    t = t.sort_values('Proj', ascending=False)
    # reset_index makes Team a real column so Pos can sit to its left.
    t = t.reset_index()
    t = t.rename(columns={'team': 'Team'})
    t.insert(0, 'Pos', range(1, len(t) + 1))

    st.dataframe(t, use_container_width=True, hide_index=True)