#!/usr/bin/env python3
import argparse, re, os
from datetime import datetime
import numpy as np
import pandas as pd
import folium

# ---- tunables ----
AM_START_MIN = 7*60 + 30    # 07:30
AM_END_MIN   = 9*60 + 30    # 09:30
PARKRIDE_NEAR_METERS  = 600.0
ALT_PARK_WITHIN_METERS = 2000.0

# ---- helpers ----
def parse_point_wkt_wgs84(s: str):
    s = str(s).strip()
    m = re.match(r"POINT\s*\(\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\)", s, flags=re.IGNORECASE)
    if m:  # WKT is (lon lat)
        lon = float(m.group(1)); lat = float(m.group(2))
        return lat, lon
    m2 = re.match(r"\s*\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?\s*$", s)
    if m2:
        lat = float(m2.group(1)); lon = float(m2.group(2))
        return lat, lon
    return np.nan, np.nan

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * R

def to_minutes_since_midnight(s):
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t): return np.nan
    return int(t.hour)*60 + int(t.minute)

# --- congestion colouring (simple, time-based) ---
def minutes_from_arg(hhmm: str | None):
    if not hhmm:  # default = now
        now = datetime.now()
        return now.hour*60 + now.minute
    hh, mm = hhmm.split(":")
    return int(hh)*60 + int(mm)

def congestion_level(mins: int) -> float:
    peak = 8*60 + 20
    width = 40
    return float(np.clip(np.exp(-((mins - peak) ** 2) / (2 * width**2)), 0, 1))

def congestion_color(level: float) -> str:
    if level < 0.33: return "#2ca25f"  # green
    if level < 0.66: return "#feb24c"  # yellow
    return "#de2d26"                   # red

# ---- bus routes overlay ----
def try_draw_bus_routes(m: folium.Map, bus_routes_path: str, max_routes: int = 80):
    import re
    try:
        df = pd.read_csv(bus_routes_path)
        geom_col = None
        for c in df.columns:
            if "geom" in c.lower() or "wkt" in c.lower() or "shape" in c.lower():
                geom_col = c; break
        if not geom_col:
            return
        def parse_multilinestring(wkt: str):
            if not isinstance(wkt, str) or "MULTILINESTRING" not in wkt.upper():
                return []
            body = wkt[wkt.find("("):].strip()
            body = body.lstrip("(").rstrip(")")
            parts = re.split(r"\)\s*,\s*\(", body)
            lines = []
            for p in parts:
                p = p.strip().lstrip("(").rstrip(")")
                coords = []
                for pair in p.split(","):
                    nums = pair.strip().split()
                    if len(nums) >= 2:
                        x = float(nums[0]); y = float(nums[1])  # lon, lat
                        coords.append((y, x))                   # lat, lon
                if coords:
                    lines.append(coords)
            return lines
        for _, r in df.head(max_routes).iterrows():
            for line in parse_multilinestring(r[geom_col]):
                folium.PolyLine(line, weight=2, opacity=0.35, color="#7f8c8d").add_to(m)
    except Exception:
        pass

# ---- OSM walking path ----
_OSM_AVAILABLE = True
try:
    import osmnx as ox
    import networkx as nx
    from shapely.geometry import LineString
except Exception:
    _OSM_AVAILABLE = False

def get_osm_walk_path(lat1, lon1, lat2, lon2, dist_m=2500, cache=True):
    """
    Returns a list of (lat, lon) tuples along the *walking network* between the two points.
    Uses osmnx graph_from_point with configurable radius (meters).
    Falls back to straight line if anything fails.
    """
    if not _OSM_AVAILABLE:
        return [(lat1, lon1), (lat2, lon2)]
    try:
        # osmnx settings
        ox.settings.use_cache = cache
        ox.settings.log_console = False

        # Build/pull a walkable graph around the midpoint
        mid_lat = (lat1 + lat2) / 2.0
        mid_lon = (lon1 + lon2) / 2.0
        G = ox.graph_from_point((mid_lat, mid_lon), dist=dist_m, network_type="walk")

        # Find nearest graph nodes
        u = ox.distance.nearest_nodes(G, lon1, lat1)
        v = ox.distance.nearest_nodes(G, lon2, lat2)

        # Shortest path by length
        route_nodes = nx.shortest_path(G, u, v, weight="length")
        # Get the geometry of the route as a list of (lat, lon)
        route_edges = ox.utils_graph.get_route_edge_attributes(G, route_nodes, attribute="geometry")
        coords = []
        for geom in route_edges:
            if geom is None:
                # Edge has no geometry → straight segment between nodes
                continue
            if isinstance(geom, LineString):
                coords.extend([(lat, lon) for lon, lat in geom.coords])  # LineString stores (lon, lat)
        # Fallback if we didn’t build coords (very short route or missing geometry)
        if not coords:
            coords = [(lat1, lon1), (lat2, lon2)]
        return coords
    except Exception:
        return [(lat1, lon1), (lat2, lon2)]

# ---- core data prep ----
def load_and_prepare(park_ride, school_buses, school_census, students_distance):
    pr = pd.read_csv(park_ride)
    pr[["lat","lon"]] = pr["Point"].apply(lambda s: pd.Series(parse_point_wkt_wgs84(s)))
    pr = pr.dropna(subset=["lat","lon"]).copy()

    sb = pd.read_csv(school_buses)
    if "Location" not in sb.columns:
        raise ValueError("Expected 'Location' WKT column in ACT_School_Bus_Services.csv")
    sb[["lat","lon"]] = sb["Location"].apply(lambda s: pd.Series(parse_point_wkt_wgs84(s)))
    sb = sb.dropna(subset=["lat","lon"]).copy()
    sb["start_min"] = sb["StartTime"].apply(to_minutes_since_midnight)
    sb["is_am_window"] = sb["start_min"].between(AM_START_MIN, AM_END_MIN, inclusive="both")
    sb["is_pm_window"] = sb["start_min"].between(15*60+30, 16*60+30, inclusive="both")

    school_col = "School Name" if "School Name" in sb.columns else None
    group_cols = ["lat","lon"] + ([school_col] if school_col else [])
    stations = (sb.groupby(group_cols, as_index=False)
                  .agg({"is_am_window":"max","is_pm_window":"max"})
                  .rename(columns={"is_am_window":"has_am_service",
                                   "is_pm_window":"has_pm_service"}))

    sc = pd.read_csv(school_census)
    sc["school_key"] = sc["School Name"].astype(str).str.lower()
    sc_tot = sc.groupby("school_key", as_index=False)["Students"].sum() \
               .rename(columns={"Students":"total_students_census"})
    if school_col:
        stations["school_key"] = stations[school_col].astype(str).str.strip().str.lower()
    else:
        stations["school_key"] = ""
    stations = stations.merge(sc_tot, on="school_key", how="left")

    sd = pd.read_csv(students_distance)
    if "School" in sd.columns:
        sd["school_key"] = sd["School"].astype(str).str.strip().str.lower()
        dist_tot = sd.groupby("school_key", as_index=False)["No. of Students"].sum() \
                     .rename(columns={"No. of Students":"total_students_distance"})
        stations = stations.merge(dist_tot, on="school_key", how="left")
    else:
        stations["total_students_distance"] = np.nan

    stations["demand_score"] = (
        stations["total_students_census"].fillna(0)*0.7 +
        stations["total_students_distance"].fillna(0)*0.3 +
        stations["has_am_service"].astype(int)*100
    )

    lat = stations["lat"].values[:, None]
    lon = stations["lon"].values[:, None]
    pr_lat = pr["lat"].values[None, :]
    pr_lon = pr["lon"].values[None, :]
    d_km = haversine_km(lat, lon, pr_lat, pr_lon)
    order = np.argsort(d_km, axis=1)
    idx1 = order[:, 0]
    idx2 = order[:, 1] if pr.shape[0] > 1 else order[:, 0]

    dist1_m = d_km[np.arange(len(stations)), idx1] * 1000.0
    dist2_m = d_km[np.arange(len(stations)), idx2] * 1000.0

    near1 = pr.iloc[idx1][["Suburb","Location","Permit Type","lat","lon"]].copy()
    near1.columns = ["nearest_pr_suburb","nearest_pr_location","nearest_pr_permit","nearest_pr_lat","nearest_pr_lon"]

    near2 = pr.iloc[idx2][["Suburb","Location","Permit Type","lat","lon"]].copy()
    near2.columns = ["alt_pr_suburb","alt_pr_location","alt_pr_permit","alt_pr_lat","alt_pr_lon"]

    stations = pd.concat([stations.reset_index(drop=True), near1.reset_index(drop=True), near2.reset_index(drop=True)], axis=1)
    stations["nearest_pr_dist_m"] = dist1_m
    stations["alt_pr_dist_m"] = dist2_m
    stations["has_pr_within_600m"] = stations["nearest_pr_dist_m"] <= PARKRIDE_NEAR_METERS

    candidates = stations[~stations["has_pr_within_600m"]].copy()
    candidates["has_alt_pr_within_2km"] = candidates["alt_pr_dist_m"] <= ALT_PARK_WITHIN_METERS
    return pr, stations, candidates, school_col

def save_csv(candidates, school_col, out_csv):
    cols = ["lat","lon","has_am_service","has_pm_service",
            "total_students_census","total_students_distance","demand_score",
            "nearest_pr_suburb","nearest_pr_location","nearest_pr_permit",
            "nearest_pr_lat","nearest_pr_lon","nearest_pr_dist_m",
            "alt_pr_suburb","alt_pr_location","alt_pr_lat","alt_pr_lon","alt_pr_dist_m",
            "has_alt_pr_within_2km"]
    if school_col:
        cols = [school_col] + cols
    candidates.sort_values("demand_score", ascending=False)[cols].to_csv(out_csv, index=False)

def save_map(pr, stations, candidates, out_html, time_hhmm=None, bus_routes_path=None, use_osm_walk=False, osm_dist=2500):
    mins = minutes_from_arg(time_hhmm)
    level = congestion_level(mins)
    m = folium.Map(location=[stations["lat"].mean(), stations["lon"].mean()], zoom_start=11)

    if bus_routes_path:
        try_draw_bus_routes(m, bus_routes_path)

    # Park & Ride markers
    for _, r in pr.iterrows():
        folium.CircleMarker((r["lat"], r["lon"]), radius=5, weight=1, fill=True, fill_opacity=0.7,
                            tooltip=f"Park & Ride: {r['Suburb']}",
                            popup=f"<b>{r['Suburb']}</b><br>{r['Location']}<br><i>{r.get('Permit Type','')}</i>",
                            color="#2c7fb8").add_to(m)

    # Stations
    school_col = "School Name" if "School Name" in stations.columns else None
    for _, r in stations.iterrows():
        color = "#2ca25f" if r["has_pr_within_600m"] else "#d95f0e"
        name = str(r[school_col]) if school_col else "Station"
        popup = (f"<b>{name}</b><br>"
                 f"AM: {bool(r['has_am_service'])}, PM: {bool(r['has_pm_service'])}<br>"
                 f"Demand score: {r['demand_score']:.0f}<br>"
                 f"P&R within 600m: {bool(r['has_pr_within_600m'])}<br>"
                 f"Nearest P&R: {r['nearest_pr_suburb']} ({r['nearest_pr_dist_m']:.0f} m)")
        folium.CircleMarker((r["lat"], r["lon"]), radius=5, weight=1, fill=True, fill_opacity=0.9,
                            tooltip=f"Station → {name}", popup=popup, color=color).add_to(m)

    # Primary & Alternative lines (with OSM walking for child segment if enabled)
    for _, r in candidates.iterrows():
        drive_color = congestion_color(level)
        # Parent drive segment: station → nearest PR (solid, congestion-coloured)
        folium.PolyLine([(r["lat"], r["lon"]), (r["nearest_pr_lat"], r["nearest_pr_lon"])],
                        weight=4, opacity=0.75, color=drive_color,
                        tooltip=f"Primary route to P&R (~{r['nearest_pr_dist_m']:.0f} m, congestion-coloured)").add_to(m)

        # CHILD WALK: nearest PR → station (OSM if enabled, else straight)
        if use_osm_walk:
            walk_coords = get_osm_walk_path(r["nearest_pr_lat"], r["nearest_pr_lon"],
                                            r["lat"], r["lon"], dist_m=osm_dist, cache=True)
        else:
            walk_coords = [(r["nearest_pr_lat"], r["nearest_pr_lon"]), (r["lat"], r["lon"])]

        folium.PolyLine(walk_coords, weight=4, opacity=0.9, color="#31a354", dash_array="6 6",
                        tooltip="Child walking path (OSM network)" if use_osm_walk else "Child walking (straight-line)").add_to(m)

        # Alternative P&R: dashed blue (still straight for MVP)
        folium.PolyLine([(r["lat"], r["lon"]), (r["alt_pr_lat"], r["alt_pr_lon"])],
                        weight=3, opacity=0.6, color="#377eb8", dash_array="8 6",
                        tooltip=f"Alternative P&R (~{r['alt_pr_dist_m']:.0f} m)").add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 6px;">
      <b>Legend</b><br>
      <svg width="140" height="12"><line x1="0" y1="6" x2="140" y2="6" stroke="{congestion_color(level)}" stroke-width="4"/></svg>
      Primary drive (congestion colour)<br>
      <svg width="140" height="12"><line x1="0" y1="6" x2="140" y2="6" stroke="#31a354" stroke-width="4" stroke-dasharray="6,6"/></svg>
      Walking (OSM)<br>
      <svg width="140" height="12"><line x1="0" y1="6" x2="140" y2="6" stroke="#377eb8" stroke-width="4" stroke-dasharray="8,6"/></svg>
      Alternative P&amp;R<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(out_html)

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park_ride", required=True)
    ap.add_argument("--school_buses", required=True)
    ap.add_argument("--school_census", required=True)
    ap.add_argument("--students_distance", required=True)
    ap.add_argument("--out_csv", default="out/recommended_candidates.csv")
    ap.add_argument("--out_map", default="out/recommendations_map.html")
    ap.add_argument("--time", default=None, help='Optional HH:MM for congestion colouring (e.g. "08:15")')
    ap.add_argument("--bus_routes", default=None, help="Optional Bus_Routes.csv (MULTILINESTRING) to overlay")
    ap.add_argument("--osm_walk", action="store_true", help="Use OSM to draw real walking path PR↔station")
    ap.add_argument("--osm_dist", type=int, default=2500, help="OSM fetch radius (meters) around midpoint")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_map), exist_ok=True)

    pr, stations, candidates, school_col = load_and_prepare(
        args.park_ride, args.school_buses, args.school_census, args.students_distance
    )
    save_csv(candidates, school_col, args.out_csv)
    save_map(
        pr, stations, candidates, args.out_map,
        time_hhmm=args.time,
        bus_routes_path=args.bus_routes,
        use_osm_walk=args.osm_walk,
        osm_dist=args.osm_dist
    )
    print("Saved:", args.out_csv)
    print("Saved:", args.out_map)

if __name__ == "__main__":
    main()
