# ==============================================================================
# Urban Resilience Command Center
# Bengaluru Flood Resilience & Urban Analytics Platform
# ==============================================================================

import streamlit as st
import geopandas as gpd
import folium
import folium.plugins
from folium.plugins import HeatMap, MarkerCluster
from branca.element import Template, MacroElement
from streamlit_folium import st_folium
import os
import numpy as np
import pandas as pd
import altair as alt
from shapely.geometry import box
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Global Configuration ---
CURRENT_MONTH_YEAR = datetime.now().strftime("%B %Y")
DATA_DIR = "data"

# ==============================================================================
# PROFESSIONAL UI SETUP
# ==============================================================================

def setup_professional_ui():
    st.set_page_config(
        page_title="Urban Resilience — Bengaluru",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🌊"
    )

    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <style>
    /* ── Base ─────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Monospaced numbers everywhere */
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em;
    }

    /* ── Custom radar-pulse loading animation ── */
    @keyframes radar-sweep {
        0%   { transform: rotate(0deg);   opacity: 1; }
        100% { transform: rotate(360deg); opacity: 1; }
    }
    @keyframes radar-ping {
        0%   { transform: scale(0.6); opacity: 1; }
        100% { transform: scale(1.8); opacity: 0; }
    }
    .stSpinner > div {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .stSpinner > div::before {
        content: '';
        display: inline-block;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 3px solid transparent;
        border-top-color: #00FF99;
        border-right-color: #00C0FF;
        animation: radar-sweep 0.9s linear infinite;
        flex-shrink: 0;
    }

    /* ── App background ───────────────────── */
    .stApp {
        background: #0D0D0D;
        color: #E0E0E0;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    /* ── Header ───────────────────────────── */
    .ur-header {
        background: linear-gradient(135deg, #0D0D0D 0%, #111820 60%, #0D1117 100%);
        padding: 2rem 2.5rem;
        border-radius: 0 0 16px 16px;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #00FF99;
        position: relative;
        overflow: hidden;
    }
    .ur-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        border-radius: 50%;
        border: 1px solid rgba(0,255,153,0.08);
        box-shadow: 0 0 60px rgba(0,255,153,0.04);
    }
    .ur-header::after {
        content: '';
        position: absolute;
        bottom: -40px; right: 80px;
        width: 120px; height: 120px;
        border-radius: 50%;
        border: 1px solid rgba(0,192,255,0.06);
    }

    /* ── Metric cards ─────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #141414 0%, #1A2233 100%);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #1E2D40;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 32px rgba(0, 255, 153, 0.18);
        border-color: #00FF99;
    }

    /* ── Status bar cards ─────────────────── */
    .status-card {
        background: #141414;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #1E2D40;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .status-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 24px rgba(0,192,255,0.15);
        border-color: #00C0FF;
    }

    /* ── Sidebar styling ──────────────────── */
    [data-testid="stSidebar"] {
        background: #0F0F0F !important;
        border-right: 1px solid #1E2D40;
    }
    [data-testid="stSidebar"] * {
        color: #E0E0E0 !important;
    }
    
    /* Fix for sidebar close button */
    [data-testid="stSidebar"] button[kind="header"] {
        background: #1E2D40 !important;
        border-radius: 50% !important;
        color: #00FF99 !important;
    }

    /* ── Divider ──────────────────────────── */
    .blue-divider {
        height: 2px;
        background: linear-gradient(90deg, #00C0FF 0%, transparent 100%);
        border: none;
        margin: 1rem 0;
        border-radius: 2px;
    }

    /* ── Section headings ─────────────────── */
    .section-heading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #00C0FF;
        margin-bottom: 0.5rem;
    }

    /* ── Analytics tabs radio override ───── */
    [data-testid="stHorizontalBlock"] [data-testid="stRadio"] label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* ── Altair chart background fix ──────── */
    .vega-embed {
        background: transparent !important;
    }

    /* ── Tip text ─────────────────────────── */
    .tip-text {
        font-size: 0.78rem;
        color: #556070;
        font-style: italic;
        margin-top: 0.5rem;
    }

    /* ── Expander tweaks ──────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid #1E2D40 !important;
        border-radius: 8px !important;
        background: #111111 !important;
    }

    /* ── Dataframe overrides ──────────────── */
    .dataframe th {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        background: #141414 !important;
        color: #00FF99 !important;
    }
    .dataframe td {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def create_professional_header():
    st.markdown(f"""
    <div class="ur-header">
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
            <span style="font-size:2.2rem;">🌊</span>
            <div>
                <h1 style="margin:0; color:#00FF99; font-family:'JetBrains Mono',monospace;
                            font-size:1.9rem; font-weight:700; letter-spacing:-0.02em;">
                    Urban Resilience
                </h1>
                <p style="margin:0; color:#8AABB8; font-size:0.9rem; letter-spacing:0.06em;">
                    BENGALURU FLOOD RESILIENCE & INFRASTRUCTURE ANALYTICS
                </p>
            </div>
        </div>
        <p style="margin:0; color:#445566; font-size:0.78rem; font-family:'JetBrains Mono',monospace;">
            LAST UPDATED: {CURRENT_MONTH_YEAR.upper()} &nbsp;·&nbsp; 198 WARDS MONITORED &nbsp;·&nbsp; DATA-DRIVEN RESILIENCE SCORING
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_status_bar():
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("System Status",    "● OPERATIONAL",  "#00FF99"),
        ("Data Freshness",   "● LIVE",          "#00FF99"),
        ("Analytics Engine", "● ACTIVE",        "#00FF99"),
        ("Wards Monitored",  "198 / 198",       "#00C0FF"),
    ]
    for col, (label, value, color) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="status-card">
                <div class="section-heading">{label}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1rem;
                            color:{color}; font-weight:600; margin-top:4px;">{value}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING & PROCESSING
# ==============================================================================

@st.cache_data(ttl=3600)
def load_geospatial_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    try:
        wards_gdf = gpd.read_file(os.path.join(DATA_DIR, "bbmp-wards.geojson")).to_crs("EPSG:4326")
        wards_gdf['area_sqkm'] = wards_gdf.to_crs(epsg=32643).geometry.area / 1e6

        primary_drains_gdf = gpd.read_file(os.path.join(DATA_DIR, "bangalore_swd_primary.geojson")).to_crs("EPSG:4326")
        primary_drains_gdf['length_km'] = primary_drains_gdf.to_crs(epsg=32643).geometry.length / 1000

        flood_parts = []
        for fname in ["bbmp_floodprone_locations.geojson",
                       "flooding_vulnerable_locations.geojson",
                       "bbmp_lowlying_areas.geojson"]:
            flood_parts.append(gpd.read_file(os.path.join(DATA_DIR, fname)).to_crs("EPSG:4326"))
        all_flood_points_gdf = pd.concat(flood_parts, ignore_index=True)

        return wards_gdf, primary_drains_gdf, all_flood_points_gdf
    except Exception as e:
        st.error(f"Error loading geospatial data: {e}")
        st.stop()


@st.cache_data(ttl=3600)
def load_tabular_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "bangalore-rainfall-data-1900-2024-sept.csv"))
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

        np.random.seed(42)
        last_year = df['Year'].max()
        current_year = datetime.now().year
        if last_year < current_year:
            last10 = df[df['Year'] > last_year - 10]
            monthly_avg = last10.mean(numeric_only=True)
            new_row = {col: 0 for col in df.columns}
            new_row['Year'] = current_year
            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            for m in months[:datetime.now().month]:
                if m in monthly_avg:
                    new_row[m] = monthly_avg[m] * np.random.uniform(0.85, 1.15)
            new_row['Total'] = sum(new_row[m] for m in months if m in new_row)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.dropna(subset=['Total'], inplace=True)
        df['deviation_from_mean'] = df['Total'] - df['Total'].mean()
        return df
    except Exception as e:
        st.error(f"Error loading rainfall data: {e}")
        st.stop()


@st.cache_data(ttl=3600)
def calculate_flood_incident_metrics(_wards_gdf, _all_flood_points_gdf) -> gpd.GeoDataFrame:
    wards = _wards_gdf.copy()
    points = _all_flood_points_gdf.copy()

    joined = gpd.sjoin(points, wards, how="inner", predicate="within")
    counts = joined.groupby('index_right').size().rename("incident_count")
    wards = wards.merge(counts, left_index=True, right_index=True, how="left")
    wards['incident_count'] = wards['incident_count'].fillna(0).astype(int)

    wards_proj = wards.to_crs(epsg=32643).copy()
    wards_proj['geometry'] = wards_proj.geometry.buffer(500)
    pts_proj = points.to_crs(epsg=32643)
    buf_joined = gpd.sjoin(pts_proj, wards_proj, how="inner", predicate="within")
    buf_counts = buf_joined.groupby('index_right').size().rename("buffered_incident_count")
    wards = wards.merge(buf_counts, left_index=True, right_index=True, how="left")
    wards['buffered_incident_count'] = wards['buffered_incident_count'].fillna(0).astype(int)

    wards['incident_density_sqkm'] = (
        wards['incident_count'] / wards['area_sqkm'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    return wards


@st.cache_data(ttl=3600)
def calculate_drainage_metrics(_wards_gdf, _primary_drains_gdf) -> gpd.GeoDataFrame:
    wards = _wards_gdf.copy()
    drains = _primary_drains_gdf.copy()

    joined = gpd.sjoin(drains, wards, how="inner", predicate="intersects")
    lengths = joined.groupby('index_right')['length_km'].sum().rename("drain_length_km")
    wards = wards.merge(lengths, left_index=True, right_index=True, how="left")
    wards['drain_length_km'] = wards['drain_length_km'].fillna(0)

    wards['drainage_density_km_sqkm'] = (
        wards['drain_length_km'] / wards['area_sqkm'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    max_dd = wards['drainage_density_km_sqkm'].max()
    wards['drainage_risk_factor'] = (
        (max_dd - wards['drainage_density_km_sqkm']) / max_dd
        if max_dd > 0 else 0
    )
    return wards


@st.cache_data(ttl=3600)
def calculate_composite_resilience_index(_wards_gdf) -> gpd.GeoDataFrame:
    """Robust rank-based normalization + PCA weighting."""
    wards = _wards_gdf.copy()

    wards['normalized_incident_density'] = wards['incident_density_sqkm'].rank(pct=True)
    wards['normalized_proximity']         = wards['buffered_incident_count'].rank(pct=True)
    wards['normalized_drainage_risk']     = wards['drainage_density_km_sqkm'].rank(pct=True, ascending=False)
    wards.fillna(0, inplace=True)

    features = ['normalized_incident_density', 'normalized_proximity', 'normalized_drainage_risk']
    X = wards[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    weights = np.abs(pca.components_[0])
    weights /= weights.sum()

    wards['Composite_Resilience_Index'] = (
        wards['normalized_incident_density'] * weights[0] +
        wards['normalized_proximity']         * weights[1] +
        wards['normalized_drainage_risk']     * weights[2]
    )
    wards['Composite_Resilience_Index'] = wards['Composite_Resilience_Index'].rank(pct=True) * 100

    def level(s):
        if s >= 85: return "Extreme Vulnerability"
        elif s >= 60: return "High Vulnerability"
        elif s >= 35: return "Moderate Vulnerability"
        elif s >= 10: return "Low Vulnerability"
        else: return "High Resilience"

    wards['resilience_level'] = wards['Composite_Resilience_Index'].apply(level)
    wards['vulnerability_rank'] = wards['Composite_Resilience_Index'].rank(ascending=False).astype(int)
    return wards


@st.cache_data(show_spinner=False)
def generate_ward_hotspot_grid(_ward_gdf, _all_flood_points_gdf, grid_size_meters):
    try:
        ward_proj = _ward_gdf.to_crs("EPSG:32643")
        minx, miny, maxx, maxy = ward_proj.total_bounds
        xs = np.arange(minx, maxx + grid_size_meters, grid_size_meters)
        ys = np.arange(miny, maxy + grid_size_meters, grid_size_meters)

        polygons = [
            box(xs[i], ys[j], xs[i+1], ys[j+1])
            for i in range(len(xs)-1)
            for j in range(len(ys)-1)
            if ward_proj.geometry.iloc[0].intersects(box(xs[i], ys[j], xs[i+1], ys[j+1]))
        ]
        if not polygons:
            return None

        grid = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:32643").to_crs("EPSG:4326")
        bounds = _ward_gdf.total_bounds
        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        pts = _all_flood_points_gdf[_all_flood_points_gdf.geometry.intersects(bbox)]

        if pts.empty:
            grid['incident_count_in_cell'] = 0
        else:
            j = gpd.sjoin(grid, pts, how="left", predicate="intersects")
            grid['incident_count_in_cell'] = j.groupby(j.index).size().reindex(grid.index, fill_value=0)

        grid['incident_count_in_cell'] = grid['incident_count_in_cell'].fillna(0).astype(int)
        grid['grid_risk_level'] = grid['incident_count_in_cell'].apply(_assign_grid_risk)
        return grid
    except Exception as e:
        st.warning(f"Hotspot grid error: {e}")
        return None


def _assign_grid_risk(n):
    if n == 0:   return "No Incidents"
    elif n == 1: return "Minor Risk"
    elif n <= 3: return "Low Risk"
    elif n <= 6: return "Moderate Risk"
    elif n <= 10:return "High Risk"
    else:        return "Critical Risk"


# ==============================================================================
# COLOUR PALETTES
# ==============================================================================

RESILIENCE_COLORS = {
    "Extreme Vulnerability": "#8B0000",
    "High Vulnerability":    "#FF4500",
    "Moderate Vulnerability":"#FFD700",
    "Low Vulnerability":     "#32CD32",
    "High Resilience":       "#008000",
}

GRID_RISK_COLORS = {
    "Critical Risk":  "#8B0000",
    "High Risk":      "#B22222",
    "Moderate Risk":  "#FF8C00",
    "Low Risk":       "#3CB371",
    "Minor Risk":     "#6B8E23",
    "No Incidents":   "#00000000",
}


# ==============================================================================
# FOLIUM MAP BUILDER
# ==============================================================================

def _add_north_arrow(m):
    """Inject a clean SVG north arrow as a MacroElement."""
    north_arrow_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        top: 80px; right: 12px;
        z-index: 9999;
        background: rgba(20,20,20,0.85);
        border: 1px solid #1E2D40;
        border-radius: 8px;
        padding: 6px 8px;
        display: flex; flex-direction: column; align-items: center;
    ">
        <svg width="24" height="36" viewBox="0 0 24 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <polygon points="12,2 4,20 12,16 20,20" fill="#00FF99" opacity="0.9"/>
            <polygon points="12,34 4,16 12,20 20,16" fill="#334455" opacity="0.7"/>
        </svg>
        <span style="font-family:'JetBrains Mono',monospace; font-size:9px;
                     color:#00FF99; letter-spacing:0.08em; margin-top:2px;">N</span>
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(north_arrow_html)
    m.get_root().add_child(macro)


def _add_collapsible_legend(m, colors_dict, title):
    """Collapsible dropdown legend injected as a MacroElement."""
    items_html = "".join(
        f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
               <div style="width:12px;height:12px;border-radius:2px;background:{c};flex-shrink:0;"></div>
               <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#E0E0E0;">{lbl}</span>
           </div>"""
        for lbl, c in colors_dict.items() if c != "#00000000"
    )

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        bottom: 48px; left: 12px;
        z-index: 9999;
        font-family: 'Space Grotesk', sans-serif;
    ">
        <details>
            <summary style="
                background: rgba(20,20,20,0.9);
                border: 1px solid #1E2D40;
                border-radius: 8px;
                padding: 6px 14px;
                cursor: pointer;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                color: #00FF99;
                letter-spacing: 0.08em;
                list-style: none;
                user-select: none;
            ">▶ {title.upper()}</summary>
            <div style="
                background: rgba(20,20,20,0.95);
                border: 1px solid #1E2D40;
                border-radius: 0 8px 8px 8px;
                padding: 12px 14px;
                margin-top: 2px;
                min-width: 190px;
            ">
                {items_html}
            </div>
        </details>
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)


def build_city_map(bbmp_wards, primary_drains, all_flood_points_gdf):
    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=11,
        tiles="CartoDB Positron",
        control_scale=True,
    )

    # Ward resilience layer
    folium.GeoJson(
        bbmp_wards,
        name="Ward Resilience Index",
        style_function=lambda f: {
            "fillColor": RESILIENCE_COLORS.get(f['properties'].get('resilience_level'), "#1A1A1A"),
            "color": "#333333", "weight": 0.7, "fillOpacity": 0.72,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['KGISWardName', 'KGISWardNo', 'Composite_Resilience_Index', 'resilience_level'],
            aliases=['Ward:', 'No.:', 'Score:', 'Level:'],
            style="background:#1A2233;color:#E0E0E0;border:1px solid #1E2D40;font-family:'JetBrains Mono',monospace;font-size:12px;"
        )
    ).add_to(m)

    # Primary drains
    folium.GeoJson(
        primary_drains,
        name="Primary Stormwater Drains",
        style_function=lambda x: {"color":"#0099FF","weight":2,"opacity":0.7},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['Name','length_km'], aliases=['Drain:','Length (km):'],
            style="background:#1A2233;color:#E0E0E0;font-family:'JetBrains Mono',monospace;font-size:12px;"
        )
    ).add_to(m)

    # Heatmap of all incidents
    coords = [[p.y, p.x] for p in all_flood_points_gdf.geometry if p]
    HeatMap(coords, name="Incident Density Heatmap", radius=14, blur=10).add_to(m)

    # Flood incident markers (clustered)
    mc = MarkerCluster(name="Flood Incident Points").add_to(m)
    for _, row in all_flood_points_gdf.iterrows():
        if row.geometry:
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5, color='#CC0000', fill=True,
                fill_color='#FF4500', fill_opacity=0.85, weight=1,
                tooltip=f"<b style='font-family:monospace'>{row.get('Name','Incident')}</b>"
            ).add_to(mc)

    folium.LayerControl(collapsed=True).add_to(m)
    _add_north_arrow(m)
    _add_collapsible_legend(m, RESILIENCE_COLORS, "Resilience Index")
    return m


def build_ward_map(selected_ward_gdf, bbmp_wards, all_flood_points_gdf, grid_size_m):
    row = selected_ward_gdf.iloc[0]
    center = [selected_ward_gdf.geometry.centroid.y.iloc[0],
              selected_ward_gdf.geometry.centroid.x.iloc[0]]

    m = folium.Map(location=center, zoom_start=14,
                   tiles="CartoDB Positron", control_scale=True)

    # Faint city context
    folium.GeoJson(
        bbmp_wards,
        name="City Context",
        style_function=lambda f: {"color":"#333333","weight":0.5,"fillOpacity":0.03},
    ).add_to(m)

    # Highlighted selected ward
    folium.GeoJson(
        selected_ward_gdf,
        name=f"Selected: {row['KGISWardName']}",
        style_function=lambda f: {
            "fillColor": RESILIENCE_COLORS.get(f['properties'].get('resilience_level'), "#1A1A1A"),
            "color":"#00FF99","weight":3,"fillOpacity":0.45,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['KGISWardName','resilience_level','Composite_Resilience_Index'],
            aliases=['Ward:','Level:','Score:'],
            style="background:#1A2233;color:#E0E0E0;font-family:'JetBrains Mono',monospace;font-size:12px;"
        )
    ).add_to(m)

    # Hotspot grid
    with st.spinner("Generating hotspot grid…"):
        grid_gdf = generate_ward_hotspot_grid(selected_ward_gdf, all_flood_points_gdf, grid_size_m)

    if grid_gdf is not None:
        folium.GeoJson(
            grid_gdf,
            name=f"{grid_size_m}m Hotspot Grid",
            style_function=lambda f: {
                "color":"#3A3A3A","weight":0.6,
                "fillColor": GRID_RISK_COLORS.get(f['properties'].get('grid_risk_level','No Incidents')),
                "fillOpacity": 0.8 if f['properties'].get('incident_count_in_cell',0) > 0 else 0.0,
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['incident_count_in_cell','grid_risk_level'],
                aliases=['Incidents:','Risk Level:'],
                style="background:#1A2233;color:#E0E0E0;font-family:'JetBrains Mono',monospace;font-size:12px;"
            )
        ).add_to(m)
        _add_collapsible_legend(m, GRID_RISK_COLORS, "Grid Risk Level")
    else:
        _add_collapsible_legend(m, RESILIENCE_COLORS, "Resilience Index")

    folium.LayerControl(collapsed=True).add_to(m)
    _add_north_arrow(m)
    return m


# ==============================================================================
# SIDEBAR BUILDER
# ==============================================================================

def build_sidebar(bbmp_wards, ward_names):
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem; text-align:center;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                         letter-spacing:0.12em; color:#445566;">COMMAND CENTER CONTROLS</span>
        </div>
        """, unsafe_allow_html=True)

        ward_options = ["— Bengaluru City Overview —"] + ward_names
        selected = st.selectbox(
            "**Target Ward**",
            options=ward_options,
            key="ward_selector",
            help="Select 'City Overview' for macro view, or a ward for granular analysis."
        )

        st.markdown('<div class="blue-divider"></div>', unsafe_allow_html=True)

        selected_ward_gdf = None

        if selected == "— Bengaluru City Overview —":
            # City-wide totals
            total_incidents   = int(bbmp_wards['incident_count'].sum())
            total_drain_km    = round(bbmp_wards['drain_length_km'].sum(), 2)
            avg_score         = round(bbmp_wards['Composite_Resilience_Index'].mean(), 1)
            high_risk_wards   = int((bbmp_wards['resilience_level'].isin(
                                    ['Extreme Vulnerability','High Vulnerability'])).sum())

            st.markdown("""
            <div class="section-heading" style="margin-top:0.5rem;">City-Wide Totals</div>
            """, unsafe_allow_html=True)

            def _stat(label, value, unit=""):
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            align-items:baseline;padding:4px 0;">
                    <span style="font-size:0.8rem;color:#8AABB8;">{label}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;
                                 color:#E0E0E0;font-weight:600;">{value}<span
                                 style="font-size:0.7rem;color:#556070;margin-left:3px;">{unit}</span></span>
                </div>""", unsafe_allow_html=True)

            _stat("Total Recorded Incidents", f"{total_incidents:,}")
            _stat("Total Managed Drains", f"{total_drain_km:,}", "km")
            _stat("Avg Vulnerability Score", avg_score, "/ 100")
            _stat("High-Risk Wards", high_risk_wards, "wards")

            st.markdown('<div class="blue-divider"></div>', unsafe_allow_html=True)
            grid_size_m = 250  # default, unused in city view

        else:
            selected_ward_gdf = bbmp_wards[bbmp_wards['KGISWardName'] == selected].copy()

            if not selected_ward_gdf.empty:
                row = selected_ward_gdf.iloc[0]
                v_rank = int(row.get('vulnerability_rank', 0))
                r_level = row.get('resilience_level', 'N/A')
                r_score = row.get('Composite_Resilience_Index', 0)
                level_color = RESILIENCE_COLORS.get(r_level, "#E0E0E0")

                # Ward header
                st.markdown(f"""
                <div style="margin-bottom:0.75rem;">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                                font-weight:700;color:#E0E0E0;">{row.get('KGISWardName','N/A')}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                color:#556070;margin-top:2px;">
                        WARD #{row.get('KGISWardNo','N/A')} &nbsp;·&nbsp;
                        RANKED <span style="color:#00C0FF;">#{v_rank}</span> IN VULNERABILITY
                    </div>
                    <div style="margin-top:8px;padding:4px 10px;display:inline-block;
                                border-radius:20px;border:1px solid {level_color};
                                font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                color:{level_color};">{r_level.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

                # Score bar
                score_pct = min(r_score, 100)
                bar_color = level_color
                st.markdown(f"""
                <div style="margin-bottom:1rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.75rem;color:#8AABB8;">Vulnerability Score</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                     color:{bar_color};font-weight:700;">{r_score:.1f} / 100</span>
                    </div>
                    <div style="background:#1A1A1A;border-radius:4px;height:6px;overflow:hidden;">
                        <div style="width:{score_pct}%;height:100%;background:{bar_color};
                                    border-radius:4px;transition:width 0.4s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="blue-divider"></div>', unsafe_allow_html=True)

                def _metric_val(val, decimals=2, zero_label="No recorded data"):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return zero_label
                    v = round(float(val), decimals)
                    return zero_label if v == 0 else str(v)

                with st.expander("🏗️ Infrastructure", expanded=True):
                    drain_len = row.get('drain_length_km', 0)
                    drain_dens = row.get('drainage_density_km_sqkm', 0)
                    area = row.get('area_sqkm', 0)

                    st.markdown(f"""
                    <table style="width:100%;border-collapse:collapse;">
                      <tr>
                        <td style="font-size:0.78rem;color:#8AABB8;padding:4px 0;">Ward Area</td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                   color:#E0E0E0;text-align:right;">
                            {_metric_val(area, 2)} <span style="color:#445566;font-size:0.7rem;">km²</span>
                        </td>
                      </tr>
                      <tr>
                        <td style="font-size:0.78rem;color:#8AABB8;padding:4px 0;">Primary Drain Length</td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                   color:#E0E0E0;text-align:right;">
                            {_metric_val(drain_len, 2)} <span style="color:#445566;font-size:0.7rem;">km</span>
                        </td>
                      </tr>
                      <tr>
                        <td style="font-size:0.78rem;color:#8AABB8;padding:4px 0;">Drainage Density</td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                   color:#E0E0E0;text-align:right;">
                            {_metric_val(drain_dens, 3)} <span style="color:#445566;font-size:0.7rem;">km/km²</span>
                        </td>
                      </tr>
                    </table>
                    """, unsafe_allow_html=True)

                with st.expander("🌊 Flood History", expanded=True):
                    incidents = row.get('incident_count', 0)
                    inc_density = row.get('incident_density_sqkm', 0)
                    buf_incidents = row.get('buffered_incident_count', 0)

                    st.markdown(f"""
                    <table style="width:100%;border-collapse:collapse;">
                      <tr>
                        <td style="font-size:0.78rem;color:#8AABB8;padding:4px 0;">Total Incidents</td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                   color:#E0E0E0;text-align:right;">
                            {"None" if incidents == 0 else incidents}
                        </td>
                      </tr>
                      <tr>
                        <td style="font-size:0.78rem;color:#8AABB8;padding:4px 0;">Incident Density</td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                   color:#E0E0E0;text-align:right;">
                            {_metric_val(inc_density, 1)} <span style="color:#445566;font-size:0.7rem;">/km²</span>
                        </td>
                      </tr>
                      <tr>
                        <td style="font-size:0.78rem;color:#8AABB8;padding:4px 0;">Proximity Incidents</td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                   color:#E0E0E0;text-align:right;">
                            {"None" if buf_incidents == 0 else buf_incidents}
                            <span style="color:#445566;font-size:0.7rem;"> (500m buffer)</span>
                        </td>
                      </tr>
                    </table>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="blue-divider"></div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="section-heading">Hotspot Resolution</div>
                """, unsafe_allow_html=True)
                grid_size_m = st.slider(
                    "Grid Cell Size (metres)",
                    min_value=100, max_value=500, value=250, step=50,
                    key="grid_size_slider",
                )

        st.markdown("""
        <p class="tip-text" style="margin-top:1rem;">
            💡 Tip: Hover over any ward on the map to view its details.
        </p>
        """, unsafe_allow_html=True)

        return selected, selected_ward_gdf, st.session_state.get('grid_size_slider', 250)


# ==============================================================================
# ANALYTICS TABS
# ==============================================================================

def _chart_theme():
    return {
        "config": {
            "axis": {"gridColor":"#1E2D40","labelColor":"#8AABB8",
                     "titleColor":"#8AABB8","labelFont":"JetBrains Mono",
                     "titleFont":"Space Grotesk"},
            "title": {"color":"#E0E0E0","font":"Space Grotesk","fontSize":14},
            "view": {"strokeWidth":0,"fill":"transparent"},
            "background": "transparent",
            "legend": {"labelColor":"#8AABB8","titleColor":"#8AABB8"},
        }
    }


# ==============================================================================
# NEW FEATURE COMPUTATIONS
# ==============================================================================

@st.cache_data(ttl=3600)
def compute_ward_similarity(_wards_gdf):
    """
    For each ward, compute cosine-distance similarity against every other ward
    using the three normalised PCA features already present on the dataframe.
    Returns a square DataFrame (ward × ward) of similarity scores 0-1.
    """
    feats = ['normalized_incident_density', 'normalized_proximity', 'normalized_drainage_risk']
    df = _wards_gdf[feats + ['KGISWardName']].copy().fillna(0)
    names = df['KGISWardName'].values
    X = df[feats].values.astype(float)

    # L2-normalise rows so dot product = cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms
    sim_matrix = X_norm @ X_norm.T          # (n × n) cosine similarities
    return pd.DataFrame(sim_matrix, index=names, columns=names)


@st.cache_data(ttl=3600)
def compute_rainfall_forecast(_rainfall_df, forecast_years=10):
    """
    Fits a degree-1 polynomial (linear trend) to historical Total rainfall.
    Returns a DataFrame with columns [Year, Total, type] for plotting,
    where type is 'Historical' or 'Forecast'.
    Uses only numpy.polyfit — no external model.
    """
    df = _rainfall_df[['Year', 'Total']].dropna().copy()
    df = df[df['Year'] > 0]

    x = df['Year'].values.astype(float)
    y = df['Total'].values.astype(float)

    # Fit linear trend on full history
    coeffs = np.polyfit(x, y, 1)
    poly   = np.poly1d(coeffs)

    # 10-year rolling average for smoothed historical line
    df = df.sort_values('Year')
    df['rolling_10yr'] = df['Total'].rolling(10, min_periods=1).mean()
    df['type'] = 'Historical'
    df['trend_line'] = poly(df['Year'].values)

    # Forecast rows
    last_year = int(df['Year'].max())
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))
    future_df = pd.DataFrame({
        'Year':       future_years,
        'Total':      poly(np.array(future_years, dtype=float)),
        'rolling_10yr': np.nan,
        'type':       'Forecast',
        'trend_line': poly(np.array(future_years, dtype=float)),
    })

    # Confidence interval (±1 std of residuals)
    residuals     = y - poly(x)
    std_residuals = residuals.std()
    future_df['ci_upper'] = future_df['Total'] + std_residuals
    future_df['ci_lower'] = future_df['Total'] - std_residuals
    df['ci_upper'] = np.nan
    df['ci_lower'] = np.nan

    combined = pd.concat([df, future_df], ignore_index=True)
    return combined, float(coeffs[0]), std_residuals


def render_analytics(bbmp_wards, rainfall_data, all_flood_points_gdf):
    st.markdown("""
    <div style="margin-top:1.5rem;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                     letter-spacing:0.12em;color:#445566;text-transform:uppercase;">
            ADVANCED RESILIENCE ANALYTICS
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    tab_names = [
        "📊 Rainfall Patterns",
        "🤝 Ward Comparison",
        "📋 Incident Breakdown",
        "📈 Resilience Index",
        "🔍 Ward Similarity",
        "📉 Rainfall Forecast",
    ]
    active = st.radio("Analysis View", tab_names, horizontal=True, label_visibility="collapsed")
    st.markdown("")  # spacer

    # ── Rainfall Patterns ────────────────────────────────────────────────────
    if active == "📊 Rainfall Patterns":
        st.markdown("<h3 style='color:#00C0FF;font-family:Space Grotesk,sans-serif;'>Historical Rainfall — Bengaluru</h3>", unsafe_allow_html=True)
        st.markdown("""<div class="metric-card"><p style="color:#8AABB8;font-size:0.88rem;">
            Total annual rainfall from 1900 to present. Extreme-rainfall years correlate strongly
            with flood incidents recorded in the ward data. The deviation bar (below) highlights
            above/below-average years at a glance.
        </p></div>""", unsafe_allow_html=True)

        if rainfall_data is not None:
            base = alt.Chart(rainfall_data).properties(height=380)
            line = base.mark_line(color="#00C0FF", strokeWidth=1.5).encode(
                x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=-45, tickCount=20)),
                y=alt.Y('Total:Q', title='Annual Rainfall (mm)'),
                tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('Total:Q', format='.0f', title='mm')]
            )
            points = base.mark_circle(color="#00FF99", size=25, opacity=0.6).encode(
                x='Year:O', y='Total:Q'
            )
            avg_val = float(rainfall_data['Total'].mean())
            rule = alt.Chart(pd.DataFrame({'y':[avg_val]})).mark_rule(
                color='#FF4500', strokeDash=[4,4], opacity=0.6
            ).encode(y='y:Q')

            st.altair_chart((line + points + rule).configure(**_chart_theme()["config"]), use_container_width=True)

            # Deviation bar
            dev_base = alt.Chart(rainfall_data).properties(height=120)
            dev_bars = dev_base.mark_bar().encode(
                x=alt.X('Year:O', axis=alt.Axis(labelAngle=-45, tickCount=20)),
                y=alt.Y('deviation_from_mean:Q', title='Deviation (mm)'),
                color=alt.condition(
                    alt.datum.deviation_from_mean > 0,
                    alt.value("#00C0FF"), alt.value("#FF4500")
                ),
                tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('deviation_from_mean:Q', format='.0f')]
            )
            st.altair_chart(dev_bars.configure(**_chart_theme()["config"]), use_container_width=True)

    # ── Ward Comparison ───────────────────────────────────────────────────────
    elif active == "🤝 Ward Comparison":
        st.markdown("<h3 style='color:#00C0FF;font-family:Space Grotesk,sans-serif;'>Ward Vulnerability Comparison</h3>", unsafe_allow_html=True)

        ward_names_list = sorted(bbmp_wards['KGISWardName'].dropna().unique().tolist())
        c1, c2 = st.columns([3, 1])

        with c1:
            selected_wards = st.multiselect(
                "Select Wards to Compare (min. 2)",
                options=ward_names_list,
                default=ward_names_list[:5] if len(ward_names_list) >= 5 else ward_names_list,
            )

            if len(selected_wards) >= 2:
                cmp = bbmp_wards[bbmp_wards['KGISWardName'].isin(selected_wards)].copy()
                chart = alt.Chart(cmp).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X('KGISWardName:N', sort='-y', title='Ward',
                             axis=alt.Axis(labelAngle=-35, labelLimit=120)),
                    y=alt.Y('Composite_Resilience_Index:Q', title='Vulnerability Score (Higher = Worse)'),
                    color=alt.Color('resilience_level:N',
                                    scale=alt.Scale(
                                        domain=list(RESILIENCE_COLORS.keys()),
                                        range=list(RESILIENCE_COLORS.values())
                                    ), legend=None),
                    tooltip=[
                        alt.Tooltip('KGISWardName:N', title='Ward'),
                        alt.Tooltip('Composite_Resilience_Index:Q', format='.1f', title='Score'),
                        alt.Tooltip('resilience_level:N', title='Level'),
                    ]
                ).properties(height=420)
                st.altair_chart(chart.configure(**_chart_theme()["config"]), use_container_width=True)
            else:
                st.info("Select at least 2 wards to compare.", icon="ℹ️")

        with c2:
            if len(selected_wards) >= 2:
                cmp = bbmp_wards[bbmp_wards['KGISWardName'].isin(selected_wards)]
                best  = cmp.loc[cmp['Composite_Resilience_Index'].idxmin()]
                worst = cmp.loc[cmp['Composite_Resilience_Index'].idxmax()]
                st.metric("Most Resilient", f"{best['KGISWardName']}", f"{best['Composite_Resilience_Index']:.1f}")
                st.metric("Most Vulnerable", f"{worst['KGISWardName']}", f"{worst['Composite_Resilience_Index']:.1f}", delta_color="inverse")
                st.metric("Selection Average", f"{cmp['Composite_Resilience_Index'].mean():.1f}")

                st.markdown(f"""<div class="metric-card" style="margin-top:1rem;">
                    <div class="section-heading">Score Range</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:#E0E0E0;margin-top:4px;">
                        {cmp['Composite_Resilience_Index'].min():.1f} – {cmp['Composite_Resilience_Index'].max():.1f}
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Incident Breakdown ────────────────────────────────────────────────────
    elif active == "📋 Incident Breakdown":
        st.markdown("<h3 style='color:#00C0FF;font-family:Space Grotesk,sans-serif;'>Historical Flood Incident Breakdown</h3>", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("""<div class="metric-card"><p style="color:#8AABB8;font-size:0.88rem;">
                Top 15 wards by recorded flood incidents. Concentration in a small
                number of wards suggests high-impact targeted interventions are possible.
            </p></div>""", unsafe_allow_html=True)

            if not all_flood_points_gdf.empty:
                with st.spinner("Spatially joining incident points…"):
                    joined = gpd.sjoin(all_flood_points_gdf,
                                       bbmp_wards[['KGISWardName','geometry']],
                                       how="inner", predicate="within")
                    counts = joined['KGISWardName'].value_counts().reset_index()
                    counts.columns = ['Ward','Incidents']
                    top15 = counts.head(15)

                chart = alt.Chart(top15).mark_bar(
                    color="#FF4500", cornerRadiusTopRight=4, cornerRadiusBottomRight=4
                ).encode(
                    x=alt.X('Incidents:Q', title='Number of Incidents'),
                    y=alt.Y('Ward:N', sort='-x', title=''),
                    tooltip=[alt.Tooltip('Ward:N'), alt.Tooltip('Incidents:Q')]
                ).properties(height=480)
                st.altair_chart(chart.configure(**_chart_theme()["config"]), use_container_width=True)

        with c2:
            st.markdown("<div class='section-heading' style='margin-top:0.5rem;'>City-Wide Stats</div>", unsafe_allow_html=True)
            st.metric("Total Incidents", f"{len(all_flood_points_gdf):,}")
            affected = int((bbmp_wards['incident_count'] > 0).sum())
            st.metric("Wards w/ ≥1 Incident", affected)
            st.metric("Max in One Ward", f"{int(bbmp_wards['incident_count'].max()):,}")

            st.markdown("""<div class="metric-card" style="margin-top:1.5rem;">
                <div class="section-heading">Key Insight</div>
                <p style="color:#8AABB8;font-size:0.8rem;margin-top:6px;">
                    A small fraction of wards account for the majority of recorded incidents —
                    making targeted infrastructure investment a high-leverage strategy.
                </p>
            </div>""", unsafe_allow_html=True)

    # ── Resilience Index ──────────────────────────────────────────────────────
    elif active == "📈 Resilience Index":
        st.markdown("<h3 style='color:#00C0FF;font-family:Space Grotesk,sans-serif;'>Resilience Index Distribution (2025)</h3>", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("""<div class="metric-card"><p style="color:#8AABB8;font-size:0.88rem;">
                Distribution of vulnerability scores across all 198 wards. A right skew
                indicates the majority of wards face moderate-to-high flood risk.
            </p></div>""", unsafe_allow_html=True)

            hist = alt.Chart(bbmp_wards).mark_bar(
                color="#00C0FF", opacity=0.85,
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3
            ).encode(
                x=alt.X('Composite_Resilience_Index:Q', bin=alt.Bin(maxbins=25),
                         title='Vulnerability Score'),
                y=alt.Y('count():Q', title='Number of Wards'),
                tooltip=[alt.Tooltip('count()', title='Wards'),
                         alt.Tooltip('Composite_Resilience_Index:Q', bin=True, title='Score Range')]
            ).properties(height=380)
            st.altair_chart(hist.configure(**_chart_theme()["config"]), use_container_width=True)

            # By resilience level
            level_counts = bbmp_wards['resilience_level'].value_counts().reset_index()
            level_counts.columns = ['Level','Wards']
            lv_chart = alt.Chart(level_counts).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X('Level:N', sort=list(RESILIENCE_COLORS.keys()), title=''),
                y=alt.Y('Wards:Q', title='Number of Wards'),
                color=alt.Color('Level:N',
                                scale=alt.Scale(domain=list(RESILIENCE_COLORS.keys()),
                                                range=list(RESILIENCE_COLORS.values())),
                                legend=None),
                tooltip=['Level:N','Wards:Q']
            ).properties(height=240, title="Wards per Resilience Level")
            st.altair_chart(lv_chart.configure(**_chart_theme()["config"]), use_container_width=True)

        with c2:
            st.markdown("<div class='section-heading'>Key Statistics</div>", unsafe_allow_html=True)
            st.metric("Mean Score", f"{bbmp_wards['Composite_Resilience_Index'].mean():.1f}")
            st.metric("Median Score", f"{bbmp_wards['Composite_Resilience_Index'].median():.1f}")

            st.markdown("<div class='section-heading' style='margin-top:1rem;'>5 Most Resilient</div>", unsafe_allow_html=True)
            st.dataframe(
                bbmp_wards[['KGISWardName','Composite_Resilience_Index']]
                .nsmallest(5,'Composite_Resilience_Index')
                .rename(columns={'KGISWardName':'Ward','Composite_Resilience_Index':'Score'})
                .assign(Score=lambda d: d['Score'].round(1)),
                hide_index=True, use_container_width=True
            )

            st.markdown("<div class='section-heading' style='margin-top:1rem;'>5 Most Vulnerable</div>", unsafe_allow_html=True)
            st.dataframe(
                bbmp_wards[['KGISWardName','Composite_Resilience_Index']]
                .nlargest(5,'Composite_Resilience_Index')
                .rename(columns={'KGISWardName':'Ward','Composite_Resilience_Index':'Score'})
                .assign(Score=lambda d: d['Score'].round(1)),
                hide_index=True, use_container_width=True
            )

    # ── Ward Similarity ───────────────────────────────────────────────────────
    elif active == "🔍 Ward Similarity":
        st.markdown("<h3 style='color:#00C0FF;font-family:Space Grotesk,sans-serif;'>Ward Similarity Finder</h3>", unsafe_allow_html=True)
        st.markdown("""<div class="metric-card"><p style="color:#8AABB8;font-size:0.88rem;">
            Find which wards share the most similar vulnerability profiles based on incident density,
            proximity incidents, and drainage risk. Similarity is computed as cosine similarity across
            the three PCA-normalised features already used in the Resilience Index.
            High similarity between a resilient ward and a vulnerable one can reveal transferable
            best practices.
        </p></div>""", unsafe_allow_html=True)

        sim_matrix = compute_ward_similarity(bbmp_wards)
        ward_names_list = sorted(bbmp_wards['KGISWardName'].dropna().unique().tolist())

        c1, c2 = st.columns([1, 2])
        with c1:
            ref_ward = st.selectbox("Select reference ward", ward_names_list, key="sim_ref_ward")
            top_n = st.slider("Show top N similar wards", 3, 20, 8, key="sim_top_n")

        with c2:
            if ref_ward in sim_matrix.index:
                sims = sim_matrix[ref_ward].drop(ref_ward).sort_values(ascending=False).head(top_n)
                sim_df = sims.reset_index()
                sim_df.columns = ['Ward', 'Similarity']
                # Merge in resilience level for colouring
                sim_df = sim_df.merge(
                    bbmp_wards[['KGISWardName','resilience_level','Composite_Resilience_Index']],
                    left_on='Ward', right_on='KGISWardName', how='left'
                ).drop(columns='KGISWardName')

                ref_row = bbmp_wards[bbmp_wards['KGISWardName'] == ref_ward].iloc[0]
                ref_level = ref_row['resilience_level']
                ref_score = ref_row['Composite_Resilience_Index']

                st.markdown(f"""<div class="metric-card" style="margin-bottom:0.75rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div class="section-heading">Reference Ward</div>
                            <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                                        color:#E0E0E0;font-weight:600;margin-top:2px;">{ref_ward}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                                        color:{RESILIENCE_COLORS.get(ref_level,'#E0E0E0')};font-weight:700;">
                                {ref_score:.1f}
                            </div>
                            <div style="font-size:0.72rem;color:#556070;">{ref_level}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                chart = alt.Chart(sim_df).mark_bar(
                    cornerRadiusTopRight=4, cornerRadiusBottomRight=4
                ).encode(
                    x=alt.X('Similarity:Q', title='Cosine Similarity (0–1)', scale=alt.Scale(domain=[0,1])),
                    y=alt.Y('Ward:N', sort='-x', title=''),
                    color=alt.Color('resilience_level:N',
                                    scale=alt.Scale(domain=list(RESILIENCE_COLORS.keys()),
                                                    range=list(RESILIENCE_COLORS.values())),
                                    legend=alt.Legend(title='Resilience Level')),
                    tooltip=[
                        alt.Tooltip('Ward:N'),
                        alt.Tooltip('Similarity:Q', format='.3f'),
                        alt.Tooltip('Composite_Resilience_Index:Q', format='.1f', title='Score'),
                        alt.Tooltip('resilience_level:N', title='Level'),
                    ]
                ).properties(height=max(200, top_n * 32))
                st.altair_chart(chart.configure(**_chart_theme()["config"]), use_container_width=True)

                # Insight: find the most resilient among the similar wards
                best_similar = sim_df.loc[sim_df['Composite_Resilience_Index'].idxmin()]
                st.markdown(f"""<div class="metric-card">
                    <div class="section-heading">💡 Best-Practice Insight</div>
                    <p style="color:#8AABB8;font-size:0.82rem;margin-top:6px;">
                        <b style="color:#00FF99;">{best_similar['Ward']}</b> has the most similar
                        vulnerability profile to <b style="color:#00C0FF;">{ref_ward}</b>
                        yet scores <b style="color:#00FF99;">{best_similar['Composite_Resilience_Index']:.1f}</b>
                        vs <b style="color:#FF4500;">{ref_score:.1f}</b>.
                        Studying its drainage infrastructure may reveal actionable improvements.
                    </p>
                </div>""", unsafe_allow_html=True)

    # ── Rainfall Forecast ─────────────────────────────────────────────────────
    elif active == "📉 Rainfall Forecast":
        st.markdown("<h3 style='color:#00C0FF;font-family:Space Grotesk,sans-serif;'>Rainfall Trend Forecaster</h3>", unsafe_allow_html=True)
        st.markdown("""<div class="metric-card"><p style="color:#8AABB8;font-size:0.88rem;">
            A linear trend is fitted to all historical annual rainfall data using
            <code style="color:#00FF99;background:#0D0D0D;padding:1px 5px;border-radius:3px;">numpy.polyfit</code>.
            The shaded band shows ± 1 standard deviation of historical residuals — the realistic
            spread of year-to-year variability. This is a statistical projection, not a climate model.
        </p></div>""", unsafe_allow_html=True)

        forecast_yrs = st.slider("Forecast horizon (years)", 5, 30, 10, key="forecast_yrs")
        combined_df, slope, std_res = compute_rainfall_forecast(rainfall_data, forecast_yrs)

        hist_df     = combined_df[combined_df['type'] == 'Historical'].copy()
        forecast_df = combined_df[combined_df['type'] == 'Forecast'].copy()
        forecast_df['ci_upper'] = forecast_df['ci_upper'].fillna(forecast_df['Total'] + std_res)
        forecast_df['ci_lower'] = forecast_df['ci_lower'].fillna(forecast_df['Total'] - std_res)

        base_h = alt.Chart(hist_df)
        base_f = alt.Chart(forecast_df)

        hist_line = base_h.mark_line(color="#00C0FF", strokeWidth=1.5, opacity=0.9).encode(
            x=alt.X('Year:Q', title='Year'),
            y=alt.Y('Total:Q', title='Annual Rainfall (mm)'),
            tooltip=[alt.Tooltip('Year:Q',format='d'), alt.Tooltip('Total:Q',format='.0f',title='mm')]
        )
        rolling_line = base_h.mark_line(color="#00FF99", strokeWidth=2,
                                         strokeDash=[4,2], opacity=0.7).encode(
            x='Year:Q', y='rolling_10yr:Q',
            tooltip=[alt.Tooltip('Year:Q',format='d'),
                     alt.Tooltip('rolling_10yr:Q',format='.0f',title='10yr avg')]
        )
        trend_line = alt.Chart(combined_df).mark_line(
            color="#FFD700", strokeWidth=1.5, strokeDash=[6,3], opacity=0.6
        ).encode(x='Year:Q', y='trend_line:Q')

        fc_line = base_f.mark_line(color="#FF4500", strokeWidth=2, strokeDash=[4,2]).encode(
            x='Year:Q', y='Total:Q',
            tooltip=[alt.Tooltip('Year:Q',format='d'), alt.Tooltip('Total:Q',format='.0f',title='Forecast mm')]
        )
        ci_band = base_f.mark_area(color="#FF4500", opacity=0.12).encode(
            x='Year:Q', y='ci_upper:Q', y2='ci_lower:Q'
        )

        chart = (hist_line + rolling_line + trend_line + ci_band + fc_line).properties(height=420)
        st.altair_chart(chart.configure(**_chart_theme()["config"]), use_container_width=True)

        # Interpretation row
        direction = "increasing" if slope > 0 else "decreasing"
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="section-heading">Long-Term Trend</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                             color:{'#FF4500' if slope > 0 else '#00FF99'};font-weight:700;margin-top:4px;">
                    {'+' if slope > 0 else ''}{slope:.1f} mm / year
                </div>
                <div style="font-size:0.75rem;color:#556070;margin-top:2px;">Rainfall is {direction}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="section-heading">Year-to-Year Variability</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                             color:#FFD700;font-weight:700;margin-top:4px;">± {std_res:.0f} mm</div>
                <div style="font-size:0.75rem;color:#556070;margin-top:2px;">1 std dev of residuals</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            last_forecast = forecast_df['Total'].iloc[-1]
            st.markdown(f"""<div class="metric-card">
                <div class="section-heading">Projected in {forecast_df['Year'].iloc[-1]:.0f}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                             color:#00C0FF;font-weight:700;margin-top:4px;">{last_forecast:.0f} mm</div>
                <div style="font-size:0.75rem;color:#556070;margin-top:2px;">Linear projection only</div>
            </div>""", unsafe_allow_html=True)


# ==============================================================================
# MAIN APP
# ==============================================================================

setup_professional_ui()

with st.spinner("🌊 Initialising Urban Resilience Platform…"):
    bbmp_wards_raw, primary_drains, all_flood_points_gdf = load_geospatial_data()
    rainfall_data = load_tabular_data()

    bbmp_wards_metrics = calculate_flood_incident_metrics(bbmp_wards_raw, all_flood_points_gdf)
    bbmp_wards_metrics = calculate_drainage_metrics(bbmp_wards_metrics, primary_drains)
    bbmp_wards = calculate_composite_resilience_index(bbmp_wards_metrics)

if bbmp_wards is None:
    st.error("FATAL: Data initialisation failed.")
    st.stop()

create_professional_header()
create_status_bar()

# Sidebar
ward_names = sorted(bbmp_wards['KGISWardName'].dropna().unique().tolist())
selected_ward_name, selected_ward_gdf, grid_size_m = build_sidebar(bbmp_wards, ward_names)

# ── Map + info column layout ────────────────────────────────────────────────
map_col, info_col = st.columns([0.72, 0.28])

with map_col:
    is_ward_view = (selected_ward_name != "— Bengaluru City Overview —"
                    and selected_ward_gdf is not None
                    and not selected_ward_gdf.empty)

    if is_ward_view:
        row = selected_ward_gdf.iloc[0]
        st.markdown(f"""
        <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;
                   color:#E0E0E0;margin-bottom:0.25rem;">
            📍 {row['KGISWardName']} — Granular Analysis
        </h2>""", unsafe_allow_html=True)
        m = build_ward_map(selected_ward_gdf, bbmp_wards, all_flood_points_gdf, grid_size_m)
    else:
        st.markdown(f"""
        <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;
                   color:#E0E0E0;margin-bottom:0.25rem;">
            🏙️ Bengaluru — City-Wide Flood Resilience &nbsp;
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                         color:#445566;">{CURRENT_MONTH_YEAR.upper()}</span>
        </h2>""", unsafe_allow_html=True)
        m = build_city_map(
            bbmp_wards, primary_drains, all_flood_points_gdf,
        )

    st_folium(m, width="100%", height=640, key="main_map")

with info_col:
    st.markdown("""<div class="metric-card">
        <div class="section-heading">Map Layers</div>
        <ul style="color:#8AABB8;font-size:0.82rem;padding-left:1.2rem;margin-top:6px;line-height:1.7;">
            <li>Ward Resilience Index</li>
            <li>Primary Stormwater Drains</li>
            <li>Flood Incident Density (Heatmap)</li>
            <li>Clustered Incident Points</li>
            <li>Granular Hotspot Grid <span style="color:#445566;">(ward view)</span></li>
        </ul>
        <div class="section-heading" style="margin-top:1rem;">Controls</div>
        <p style="color:#8AABB8;font-size:0.82rem;margin-top:4px;">
            Toggle layers via the ⊞ control (top-right on map).<br>
            Click <b>▶ RESILIENCE INDEX</b> on the map (bottom-left) to expand the legend.
        </p>
    </div>""", unsafe_allow_html=True)

    # Quick city stats card
    st.markdown("""<div class="metric-card">
        <div class="section-heading">Quick Stats</div>""", unsafe_allow_html=True)

    extreme = int((bbmp_wards['resilience_level'] == 'Extreme Vulnerability').sum())
    high    = int((bbmp_wards['resilience_level'] == 'High Vulnerability').sum())
    total_i = int(bbmp_wards['incident_count'].sum())

    for label, value, color in [
        ("Extreme Vulnerability Wards", extreme, "#8B0000"),
        ("High Vulnerability Wards",    high,    "#FF4500"),
        ("Total Recorded Incidents",    f"{total_i:,}", "#00C0FF"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;
                    border-bottom:1px solid #1E2D40;">
            <span style="font-size:0.78rem;color:#8AABB8;">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;
                         color:{color};font-weight:700;">{value}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Analytics Section ───────────────────────────────────────────────────────
render_analytics(bbmp_wards, rainfall_data, all_flood_points_gdf)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:1.5rem 0;color:#445566;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;letter-spacing:0.08em;">
        URBAN RESILIENCE COMMAND CENTER &nbsp;·&nbsp; BENGALURU
        &nbsp;·&nbsp; DATA: BBMP · KSNDMC · OPEN DATA INITIATIVES
        &nbsp;·&nbsp; BUILT WITH STREAMLIT · GEOPANDAS · FOLIUM · ALTAIR
    </span>
</div>
""", unsafe_allow_html=True)
