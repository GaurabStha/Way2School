
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import re

# -----------------------
# App Setup
# -----------------------
st.set_page_config(page_title="Way2School", page_icon="🚌", layout="wide")

# -----------------------
# Sample Data To Simulate Real Inputs
# -----------------------
SCHOOLS = [
    {"name": "Civic Secondary", "lat": -35.2809, "lon": 149.1300},
    {"name": "Woden Primary", "lat": -35.3494, "lon": 149.0850},
    {"name": "Belconnen High", "lat": -35.2389, "lon": 149.0670},
    {"name": "Molonglo College", "lat": -35.3140, "lon": 149.0430},
]

PARK_AND_RIDE = [
    {"name": "Woden P&R", "lat": -35.3440, "lon": 149.0900, "capacity": 300, "base_fill": 0.55},
    {"name": "Belconnen P&R", "lat": -35.2360, "lon": 149.0700, "capacity": 280, "base_fill": 0.50},
    {"name": "Gungahlin P&R", "lat": -35.1850, "lon": 149.1360, "capacity": 220, "base_fill": 0.45},
    {"name": "Molonglo P&R (Future)", "lat": -35.3145, "lon": 149.0400, "capacity": 200, "base_fill": 0.35},
]

LIGHT_RAIL_STOPS = [
    {"name": "Gungahlin Place", "lat": -35.1855, "lon": 149.1365},
    {"name": "Dickson Interchange", "lat": -35.2600, "lon": 149.1380},
    {"name": "City", "lat": -35.2810, "lon": 149.1280},
    {"name": "Woden (Future)", "lat": -35.3450, "lon": 149.0855},
]

BUS_LINES = [
    {"name": "Bus 200 Rapid", "freq_min": 7, "avg_speed_kmh": 28},
    {"name": "Bus 300 Rapid", "freq_min": 6, "avg_speed_kmh": 30},
    {"name": "Bus 41 Local", "freq_min": 15, "avg_speed_kmh": 22},
]

INCIDENTS = [
    {"where": "Parkes Way", "type": "Roadworks", "severity": 2},
    {"where": "Cotter Rd", "type": "Congestion", "severity": 3},
    {"where": "Northbourne Ave", "type": "Minor crash cleared", "severity": 1},
]

park_ride = pd.read_csv("datasets/Park_And_Ride_Locations.csv")
def parse_point(s):
    m = re.match(r"\s*\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?\s*", str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None
park_ride[["lat","lon"]] = park_ride["Point"].apply(lambda s: pd.Series(parse_point(s)))

center_lat = park_ride["lat"].mean()
center_lon = park_ride["lon"].mean()

def now_local():
    # Adjust to user-local in real deployment;
    return datetime.now()

def time_to_minutes(t: dtime):
    return t.hour * 60 + t.minute

def predicted_fill_ratio(base_fill, dt: datetime):
    """
    Simple demand curve that rises towards 8:30–9:00 and decays after.
    Returns a float in [0, 1.1] (can slightly exceed 1 to show 'full + overflow' risk).
    """
    minutes = dt.hour * 60 + dt.minute
    peak = 8 * 60 + 40  # 8:40
    width = 70  # spread around peak
    peak_bump = np.exp(-((minutes - peak) ** 2) / (2 * (width ** 2)))
    # weekday factor (higher Mon-Thu)
    weekday = dt.weekday()
    weekday_factor = 1.0 if weekday < 4 else 0.85  # Fri a bit less
    ratio = base_fill * 0.8 + 0.45 * peak_bump * weekday_factor
    return max(0.0, min(1.1, ratio))

def predict_spaces(capacity, base_fill, dt: datetime):
    ratio = predicted_fill_ratio(base_fill, dt)
    spaces_left = int(capacity * (1 - ratio))
    return max(0, spaces_left), ratio

def mock_walk_minutes(distance_km):
    return int(12 * distance_km + 3)  # 12 min/km + 3 min buffer

def mock_bike_minutes(distance_km):
    return int(4 * distance_km + 2)   # 15 km/h approx with buffer

def mock_drive_minutes(distance_km):
    return int(3 * distance_km + 4)   # ~20 km/h urban w/ signals (demo-friendly)

def mock_light_rail_minutes(stops):
    return int(stops * 3 + 5)         # dwell time + travel

def mock_bus_minutes(distance_km, line_name):
    speed = next(l["avg_speed_kmh"] for l in BUS_LINES if l["name"] == line_name)
    return int(distance_km / max(1, speed) * 60 + 5)

def headway_penalty(line_name):
    freq = next(l["freq_min"] for l in BUS_LINES if l["name"] == line_name)
    # expected wait ~ half headway
    return int(freq / 2)

def route_safety_score(night: bool, rain: bool, incidents_nearby: int, cctv: bool, school_zone_active: bool):
    score = 86
    if night: score -= 10
    if rain: score -= 8
    score -= 4 * incidents_nearby
    if cctv: score += 6
    if school_zone_active: score += 4
    return int(max(0, min(100, score)))

def safety_badge(score):
    if score >= 80: return "🟢 Safe"
    if score >= 60: return "🟡 Caution"
    return "🔴 Avoid"

def congestion_projection(area_name: str):
    """Return congestion score by 5-min bucket 7:00–9:30 for the selected area."""
    start = datetime.combine(now_local().date(), dtime(7, 0))
    times = [start + timedelta(minutes=5 * i) for i in range(31)]  # 7:00–9:30
    base = np.array([np.exp(-((i-16) ** 2) / (2 * 18 ** 2)) for i in range(len(times))])  # peak ~8:20
    # Area-specific multipliers (Molonglo higher during growth)
    area_factor = 1.25 if "Molonglo" in area_name else 1.0
    noise = np.random.normal(0, 0.05, size=len(times))
    score = np.clip((0.35 + 0.55 * base) * area_factor + noise, 0, 1)
    return times, score

def load_candidates(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        # Normalise column names we care about if present
        cols = {c.lower(): c for c in df.columns}
        # Common names produced by the recommender script
        school_col = cols.get("school name") or cols.get("school")  # may be missing
        near_dist = cols.get("nearest_pr_dist_m")
        near_suburb = cols.get("nearest_pr_suburb")
        near_loc = cols.get("nearest_pr_location")
        near_lat = cols.get("nearest_pr_lat")
        near_lon = cols.get("nearest_pr_lon")
        lat_col = cols.get("lat")
        lon_col = cols.get("lon")

        # convenience: computed display cols if present
        if near_dist and near_dist in df.columns:
            df["Nearest P&R (m)"] = (df[near_dist]).round(0).astype("Int64")
            df["Nearest P&R (km)"] = (df[near_dist] / 1000.0).round(2)
        df.attrs["school_col"] = school_col
        df.attrs["lat_col"] = lat_col
        df.attrs["lon_col"] = lon_col
        df.attrs["near_lat"] = near_lat
        df.attrs["near_lon"] = near_lon
        return df
    except Exception as e:
        st.warning(f"Could not load candidates CSV: {e}")
        return None

if "reservations" not in st.session_state:
    st.session_state["reservations"] = {}

st.sidebar.title("Way2School")
home_area = st.sidebar.selectbox("Home area", ["Woden", "Belconnen", "Gungahlin", "Molonglo"], index=3)
school = st.sidebar.selectbox("Destination school", [s["name"] for s in SCHOOLS], index=3)
depart_time = st.sidebar.time_input("Planned departure time", value=dtime(8, 5))
weather = st.sidebar.selectbox("Weather", ["Clear", "Light Rain", "Heavy Rain"], index=0)
want_bike = st.sidebar.checkbox("Include bike options", value=False)
show_advanced = st.sidebar.checkbox("Advanced options", value=False)

suburbs = ["All"] + sorted(park_ride["Suburb"].dropna().unique().tolist())
sel_suburb = st.sidebar.selectbox("Filter by suburb", suburbs)
if sel_suburb != "All":
    park_ride = park_ride[park_ride["Suburb"] == sel_suburb]

st.sidebar.markdown("---")
map_html_path = st.sidebar.text_input(
    "P&R map (HTML) path",
    value="out/recommendations_map.html"  # change if yours is elsewhere
)
candidates_csv_path = st.sidebar.text_input(
    "P&R candidates (CSV) path",
    value="out/recommended_candidates.csv"  # change if yours is elsewhere
)

colA, colB = st.columns([3, 2])
with colA:
    st.title("🚌 Way2School")
    st.subheader("Fast, safe, flexible school transport for families")
with colB:
    tnow = now_local()
    st.metric("Today", tnow.strftime("%A, %d %b %Y"))
    st.metric("Current time", tnow.strftime("%I:%M %p"))

st.markdown("---")

# -----------------------
# Top: Morning Brief (solves Morning Chaos + Information Scatter)
# -----------------------
st.header("🌅 Morning Brief")
c1, c2, c3, c4 = st.columns(4)

# Pick a P&R near the chosen home_area for the brief
def pick_pr(area):
    mapping = {
        "Woden": "Woden P&R",
        "Belconnen": "Belconnen P&R",
        "Gungahlin": "Gungahlin P&R",
        "Molonglo": "Molonglo P&R (Future)"
    }
    target = mapping.get(area, "Woden P&R")
    return next(p for p in PARK_AND_RIDE if p["name"] == target)

pr = pick_pr(home_area)
dt = datetime.combine(now_local().date(), depart_time)
spaces, ratio = predict_spaces(pr["capacity"], pr["base_fill"], dt)

with c1:
    st.metric(f"{pr['name']} – spaces left at {depart_time.strftime('%H:%M')}", spaces)
with c2:
    pred_in_20, _ = predict_spaces(pr["capacity"], pr["base_fill"], dt + timedelta(minutes=20))
    st.metric("In 20 min", pred_in_20, delta=pred_in_20 - spaces)
with c3:
    rush_note = "High risk of full" if ratio > 0.95 else ("Filling fast" if ratio > 0.8 else "Comfortable")
    st.metric("Risk", rush_note)
with c4:
    st.button("🔒 Reserve 15-min spot", key="reserve_top", help="Demo reservation – holds a space for 15 minutes")
    if st.session_state.get("reserve_top_clicked", False):
        pass

# Reservation logic: when any reserve button is pressed, mark a reservation
def do_reserve(lot_name, when: datetime):
    key = f"{lot_name}_{when.strftime('%H%M')}"
    st.session_state["reservations"][key] = {"lot": lot_name, "until": when + timedelta(minutes=15)}

if st.button(f"Reserve at {pr['name']} for {depart_time.strftime('%H:%M')}", key="reserve_main"):
    do_reserve(pr["name"], dt)
    st.success(f"Reserved a spot at {pr['name']} until {(dt + timedelta(minutes=15)).strftime('%H:%M')}. (Demo)")


# -----------------------
# Tabs: Planner, Park&Ride, Safety, Growth, Parent
# -----------------------
tabs = st.tabs(["🧭 Planner", "🅿️ Park & Ride", "🛡️ Safety", "📈 Growth (Molonglo)", "👨‍👩‍👧 Parent"])

# -----------------------
# Tab 1: Planner – Route Confusion & Safety Anxiety
# -----------------------
with tabs[0]:
    st.subheader("Smart Route Planner")
    st.caption("Shows fastest & safest multi‑modal options with live Park & Ride and predicted waits.")
    # Mock 3 route options
    # Route A: Drive -> P&R -> Light Rail -> Walk
    lr_stops = 4 if home_area == "Gungahlin" else 2
    routeA = {
        "label": "Drive → Park&Ride → Light Rail → Walk",
        "legs": [
            ("Drive to P&R", mock_drive_minutes(4.5)),
            ("Wait light rail", 3),
            (f"Light rail ({lr_stops} stops)", mock_light_rail_minutes(lr_stops)),
            ("Walk to school", mock_walk_minutes(0.6)),
        ],
        "cost": 3.0,
        "uses_pr": True,
        "uses_bus": False,
        "uses_lr": True
    }
    # Route B: Local Bus → Rapid Bus → Walk
    routeB = {
        "label": "Local Bus → Walk",
        "legs": [
            ("Wait Bus 41", headway_penalty("Bus 41 Local")),
            ("Bus 41 (2.5 km)", mock_bus_minutes(2.5, "Bus 41 Local")),
            ("Wait Bus 300", headway_penalty("Bus 300 Rapid")),
            ("Bus 300 (5.5 km)", mock_bus_minutes(5.5, "Bus 300 Rapid")),
            ("Walk to school", mock_walk_minutes(0.4)),
        ],
        "cost": 2.5,
        "uses_pr": False,
        "uses_bus": True,
        "uses_lr": False
    }
    # Route C: Bike → Light Rail (bike onboard) → Walk (or Bike) – shown if user wants bike
    routeC = {
        "label": "Bike → Light Rail → Walk",
        "legs": [
            ("Bike to LR stop (1.6 km)", mock_bike_minutes(1.6)),
            ("Wait light rail", 4),
            ("Light rail (3 stops)", mock_light_rail_minutes(3)),
            ("Walk/Bike to school", mock_walk_minutes(0.5)),
        ],
        "cost": 0.0,
        "uses_pr": False,
        "uses_bus": False,
        "uses_lr": True
    }
    routes = [routeA, routeB] + ([routeC] if want_bike else [])
    
    # Compute safety for each route
    is_night = depart_time < dtime(6, 30)
    is_rain = weather != "Clear"
    incidents_nearby = sum(1 for inc in INCIDENTS if inc["severity"] >= 2)  # simple demo rule
    school_zone_active = dtime(8, 0) <= depart_time <= dtime(9, 30)
    
    def total_minutes(legs): return sum(m for _, m in legs)
    
    cards = st.columns(len(routes))
    for i, r in enumerate(routes):
        with cards[i]:
            tmin = total_minutes(r["legs"])
            score = route_safety_score(is_night, is_rain, incidents_nearby, cctv=True, school_zone_active=school_zone_active)
            badge = safety_badge(score)
            st.markdown(f"### {r['label']}")
            st.metric("ETA", f"{tmin} min")
            st.metric("Safety", f"{badge} ({score})")
            if r["uses_pr"]:
                s, _ = predict_spaces(pr["capacity"], pr["base_fill"], dt)
                st.metric("P&R spaces at depart", s)
            cost = "Free" if r["cost"] == 0 else f"${r['cost']:.2f}"
            st.metric("Approx. fare", cost)
            with st.expander("Leg details"):
                for leg, mins in r["legs"]:
                    st.write(f"• {leg}: {mins} min")
            if st.button(f"Start {i+1}", key=f"start_{i}"):
                st.success(f"Navigation started for: {r['label']} (demo)")
        

    st.markdown("---")
    st.subheader("Park & Ride near your destination")

    col_map, col_info = st.columns([3, 2])

    with col_map:
        # Embed the pre-built HTML map from your recommender
        try:
            with open(map_html_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=600, scrolling=True)
        except Exception as e:
            st.error(f"Couldn't load map HTML at '{map_html_path}': {e}")
            st.caption("Tip: run the P&R recommender script first to generate recommendations_map.html")

    with col_info:
        cand_df = load_candidates(candidates_csv_path)
        if cand_df is not None and len(cand_df) > 0:
            school_col_name = cand_df.attrs.get("school_col")

            # Filter by the currently selected school, if the column exists
            if school_col_name and school_col_name in cand_df.columns:
                df_school = cand_df[cand_df[school_col_name].astype(str).str.strip() == school]
                if df_school.empty:
                    st.info("No station rows found for this school in the candidates file. Showing overall top results.")
                    df_show = cand_df.sort_values("demand_score", ascending=False).head(10)
                else:
                    df_show = df_school.sort_values("Nearest P&R (m)" if "Nearest P&R (m)" in df_school.columns else "nearest_pr_dist_m").head(5)
            else:
                # Fallback: just top by demand if we can't match school names
                df_show = cand_df.sort_values("demand_score", ascending=False).head(10)

            # Friendly columns to display if present
            show_cols = []
            for c in ["School Name", "School", "Nearest P&R (m)", "Nearest P&R (km)",
                      "nearest_pr_suburb", "nearest_pr_location", "demand_score"]:
                if c in df_show.columns:
                    show_cols.append(c)

            st.markdown("**Nearest Park & Ride options**")
            st.dataframe(df_show[show_cols], use_container_width=True)

            # Little explainer
            st.caption("Distances are straight-line from station to Park & Ride in this MVP. "
                       "For production, replace with walking-network distance (OSM) and show live bus options.")
        else:
            st.info("Upload or generate a candidates CSV to list nearest P&R per station/school.")


    st.markdown("---")
    st.caption("Tip: Change weather or departure time in the sidebar to see safety/ETA adjust.")

# -----------------------
# Tab 2: Park & Ride – Morning Chaos
# -----------------------
with tabs[1]:
    st.subheader("Live Park & Ride Availability & Reservations")
    # Table of current and +20 min predictions
    rows = []
    for lot in PARK_AND_RIDE:
        now_spaces, now_ratio = predict_spaces(lot["capacity"], lot["base_fill"], dt)
        fut_spaces, fut_ratio = predict_spaces(lot["capacity"], lot["base_fill"], dt + timedelta(minutes=20))
        rows.append({
            "Lot": lot["name"],
            "Capacity": lot["capacity"],
            f"Spaces @ {depart_time.strftime('%H:%M')}": now_spaces,
            "Risk now": "High" if now_ratio>0.95 else ("Medium" if now_ratio>0.8 else "Low"),
            "Spaces in 20 min": fut_spaces,
            "Risk in 20": "High" if fut_ratio>0.95 else ("Medium" if fut_ratio>0.8 else "Low"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    for _, row in park_ride.iterrows():
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            popup_txt = f"<b>{row.get('Suburb','')}</b><br>{row.get('Location','')}<br><i>{row.get('Permit Type','')}</i>"
            folium.Marker([row["lat"], row["lon"]], tooltip=f"Park & Ride: {row.get('Suburb','')}", popup=popup_txt).add_to(m)

    st_data = st_folium(m, width=None, height=600)
    
    # Reserve dropdown
    lots_names = [l["name"] for l in PARK_AND_RIDE]
    lot_choice = st.selectbox("Choose lot to reserve", lots_names, index=lots_names.index(pr["name"]))
    if st.button("Reserve 15‑min hold", key="reserve_tab"):
        do_reserve(lot_choice, dt)
        st.success(f"Reserved at {lot_choice} until {(dt + timedelta(minutes=15)).strftime('%H:%M')} (demo).")
    
    # Show active reservations
    if st.session_state["reservations"]:
        st.markdown("#### Active Reservations")
        for k, v in st.session_state["reservations"].items():
            st.write(f"• {v['lot']} held until {v['until'].strftime('%H:%M')}")
    else:
        st.info("No active reservations.")

# -----------------------
# Tab 3: Safety – Safety Anxiety
# -----------------------
with tabs[2]:
    st.subheader("Route Safety Insights")
    st.write("Each route gets a **Safety Score** (0–100) using lighting, CCTV, incidents and conditions.")
    st.write("This page breaks down the factors so parents know *why* one route is safer than another.")
    
    # Show current incident feed (mocked)
    st.markdown("**Live Incidents (demo)**")
    inc_df = pd.DataFrame(INCIDENTS)
    st.table(inc_df)
    
    # Score explainer
    night = "Yes" if is_night else "No"
    rain = "Yes" if is_rain else "No"
    st.markdown("##### Current factors")
    st.write(f"- Night time: **{night}**")
    st.write(f"- Raining: **{rain}**")
    st.write(f"- Incidents with severity ≥2 nearby: **{incidents_nearby}**")
    st.write(f"- CCTV on route: **Yes**")
    st.write(f"- School zone active: **{'Yes' if school_zone_active else 'No'}**")
    
    score = route_safety_score(is_night, is_rain, incidents_nearby, cctv=True, school_zone_active=school_zone_active)
    st.metric("Safety score for top suggestion right now", score, help="Higher is better")
    st.progress(score/100.0)

# -----------------------
# Tab 4: Growth – Growth Pressure (Molonglo)
# -----------------------
with tabs[3]:
    st.subheader("Growth Pressure – Molonglo AM Peak Projection")
    st.caption("Predictive congestion (demo) for 7:00–9:30. Use to plan staggered departures and alternate P&R.")
    
    areas = ["Molonglo", "Woden", "Belconnen", "Gungahlin"]
    area = st.selectbox("Area", areas, index=0)
    times, score = congestion_projection(area)
    chart_df = pd.DataFrame({"time": times, "congestion": score})
    st.line_chart(chart_df, x="time", y="congestion", height=260)
    peak_row = chart_df.iloc[chart_df["congestion"].idxmax()]
    st.info(f"Peak congestion at **{peak_row['time'].strftime('%H:%M')}**. Consider departing 10–15 minutes earlier or using Park & Ride suggestion on the Planner tab.")

# -----------------------
# Tab 5: Parent – Real-time Tracking (demo)
# -----------------------
with tabs[4]:
    st.subheader("Parent Dashboard – Live Trip (Demo)")
    st.caption("Simulates your child's progress to school and ETA updates.")
    route_progress = st.slider("Trip progress", 0, 100, 20, help="Move slider to simulate travel progress")
    # Interpolate location between home_area rough coords and school coords
    area_coords = {
        "Woden": (-35.3440, 149.0900),
        "Belconnen": (-35.2360, 149.0700),
        "Gungahlin": (-35.1850, 149.1360),
        "Molonglo": (-35.3145, 149.0400)
    }
    start_lat, start_lon = area_coords[home_area]
    school_lat = next(s['lat'] for s in SCHOOLS if s['name'] == school)
    school_lon = next(s['lon'] for s in SCHOOLS if s['name'] == school)
    lat = start_lat + (school_lat - start_lat) * (route_progress/100)
    lon = start_lon + (school_lon - start_lon) * (route_progress/100)
    st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))
    remaining_min = max(0, 30 - int(route_progress/100 * 30))
    st.metric("ETA to school", f"{remaining_min} min")