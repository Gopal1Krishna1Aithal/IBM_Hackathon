# ==============================================================================
# HAURCC - Hyper-Analytical Urban Resilience Command Center (v2.0)
# Developed for Urban Resilience & Flood Management in Bengaluru
# ==============================================================================

import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
import numpy as np
import pandas as pd
import altair as alt
from shapely.geometry import box, shape, Point, MultiLineString
import branca.colormap as cm
import json
import base64
import math
from typing import Dict, Any, Tuple

# --- Global Configuration & Paths ---
CURRENT_MONTH_YEAR = "July 2025"
DATA_DIR = "data"

# ==============================================================================
# HELPER FUNCTIONS - DATA PROCESSING AND METRICS
# ==============================================================================

@st.cache_data(ttl=3600)  # Cache data for 1 hour to optimize performance
def load_geospatial_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Loads all core geospatial data files."""
    try:
        # BBMP Wards (Polygons)
        wards_path = os.path.join(DATA_DIR, "bbmp-wards.geojson")
        wards_gdf = gpd.read_file(wards_path)
        if wards_gdf.crs is None or wards_gdf.crs.is_projected:
            wards_gdf = wards_gdf.to_crs("EPSG:4326")
        
        # Calculate ward area in square kilometers
        wards_gdf_proj_area = wards_gdf.to_crs(epsg=32643)
        wards_gdf['area_sqkm'] = wards_gdf_proj_area.geometry.area / 10**6

        # Primary Drains Data
        drains_path = os.path.join(DATA_DIR, "bangalore_swd_primary.geojson")
        primary_drains_gdf = gpd.read_file(drains_path)
        if primary_drains_gdf.crs is None or primary_drains_gdf.crs.is_projected:
            primary_drains_gdf = primary_drains_gdf.to_crs("EPSG:4326")
        
        # Calculate drain lengths in km
        primary_drains_gdf_proj = primary_drains_gdf.to_crs(epsg=32643)
        primary_drains_gdf['length_km'] = primary_drains_gdf_proj.geometry.length / 1000

        # All Flood Incident Points
        floodprone_path = os.path.join(DATA_DIR, "bbmp_floodprone_locations.geojson")
        vulnerable_path = os.path.join(DATA_DIR, "flooding_vulnerable_locations.geojson")
        lowlying_path = os.path.join(DATA_DIR, "bbmp_lowlying_areas.geojson")

        floodprone_gdf = gpd.read_file(floodprone_path)
        vulnerable_gdf = gpd.read_file(vulnerable_path)
        lowlying_gdf = gpd.read_file(lowlying_path)

        for gdf in [floodprone_gdf, vulnerable_gdf, lowlying_gdf]:
            if gdf.crs is None or gdf.crs.is_projected:
                gdf = gdf.to_crs("EPSG:4326")

        all_flood_points_gdf = pd.concat([
            floodprone_gdf,
            vulnerable_gdf,
            lowlying_gdf
        ], ignore_index=True)
        
        return wards_gdf, primary_drains_gdf, all_flood_points_gdf
    
    except Exception as e:
        st.error(f"Error loading geospatial data: {e}")
        st.stop()


@st.cache_data(ttl=3600)
def load_tabular_data() -> pd.DataFrame:
    """Loads and preprocesses rainfall data."""
    try:
        rainfall_csv_path = os.path.join(DATA_DIR, "bangalore-rainfall-data-1900-2024-sept.csv")
        rainfall_df = pd.read_csv(rainfall_csv_path)
        rainfall_df['Year'] = pd.to_numeric(rainfall_df['Year'], errors='coerce').fillna(0).astype(int)
        rainfall_df.dropna(subset=['Total'], inplace=True)
        
        avg_annual_rainfall = rainfall_df['Total'].mean()
        rainfall_df['deviation_from_mean'] = rainfall_df['Total'] - avg_annual_rainfall
        
        return rainfall_df
        
    except Exception as e:
        st.error(f"Error loading rainfall data: {e}")
        st.stop()

@st.cache_data(ttl=3600)
def calculate_flood_incident_metrics(_wards_gdf, _all_flood_points_gdf) -> gpd.GeoDataFrame:
    """
    Calculates direct and proximity-based flood incident counts.
    
    Args:
        _wards_gdf (gpd.GeoDataFrame): GeoDataFrame of BBMP wards.
        _all_flood_points_gdf (gpd.GeoDataFrame): GeoDataFrame of all flood incidents.
    
    Returns:
        gpd.GeoDataFrame: Wards GeoDataFrame with new incident metrics.
    """
    wards_gdf = _wards_gdf.copy()
    all_flood_points_gdf = _all_flood_points_gdf.copy()

    # Calculate direct incident count (points within ward)
    wards_with_points = gpd.sjoin(all_flood_points_gdf, wards_gdf, how="inner", predicate="within")
    incident_counts = wards_with_points.groupby('index_right').size().rename("incident_count")
    wards_gdf = wards_gdf.merge(incident_counts, left_index=True, right_index=True, how="left")
    wards_gdf['incident_count'] = wards_gdf['incident_count'].fillna(0).astype(int)

    # Calculate proximity incident count (points within a 500m buffer)
    wards_gdf_proj_buffer = wards_gdf.to_crs(epsg=32643) 
    buffered_wards_gdf_proj = wards_gdf_proj_buffer.copy()
    buffered_wards_gdf_proj['geometry'] = buffered_wards_gdf_proj.geometry.buffer(500)

    points_proj = all_flood_points_gdf.to_crs(epsg=32643)
    wards_with_buffered_points = gpd.sjoin(points_proj, buffered_wards_gdf_proj, how="inner", predicate="within")
    buffered_incident_counts = wards_with_buffered_points.groupby('index_right').size().rename("buffered_incident_count")

    wards_gdf = wards_gdf.merge(buffered_incident_counts, left_index=True, right_index=True, how="left")
    wards_gdf['buffered_incident_count'] = wards_gdf['buffered_incident_count'].fillna(0).astype(int)
    
    # Incident Density
    wards_gdf['incident_density_sqkm'] = wards_gdf.apply(
        lambda row: (row['incident_count'] / row['area_sqkm']) if row['area_sqkm'] > 0 else 0, axis=1
    )
    wards_gdf['incident_density_sqkm'] = wards_gdf['incident_density_sqkm'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return wards_gdf


@st.cache_data(ttl=3600)
def calculate_drainage_metrics(_wards_gdf, _primary_drains_gdf) -> gpd.GeoDataFrame:
    """
    Calculates drainage-related metrics for each ward.
    
    Args:
        _wards_gdf (gpd.GeoDataFrame): GeoDataFrame of BBMP wards.
        _primary_drains_gdf (gpd.GeoDataFrame): GeoDataFrame of primary drains.
    
    Returns:
        gpd.GeoDataFrame: Wards GeoDataFrame with new drainage metrics.
    """
    wards_gdf = _wards_gdf.copy()
    primary_drains_gdf = _primary_drains_gdf.copy()
    
    # Spatial join drains to wards
    wards_with_drains = gpd.sjoin(primary_drains_gdf, wards_gdf, how="inner", predicate="intersects")
    
    # Group by ward and sum drain lengths within each ward
    drain_lengths_per_ward = wards_with_drains.groupby('index_right')['length_km'].sum().rename("drain_length_km")
    
    # Merge back to wards_gdf
    wards_gdf = wards_gdf.merge(drain_lengths_per_ward, left_index=True, right_index=True, how="left")
    wards_gdf['drain_length_km'] = wards_gdf['drain_length_km'].fillna(0)

    # Calculate Drainage Density
    wards_gdf['drainage_density_km_sqkm'] = wards_gdf.apply(
        lambda row: (row['drain_length_km'] / row['area_sqkm']) if row['area_sqkm'] > 0 else 0, axis=1
    )
    wards_gdf['drainage_density_km_sqkm'] = wards_gdf['drainage_density_km_sqkm'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate an inverse drainage risk factor: lower density = higher risk
    max_drainage_density = wards_gdf['drainage_density_km_sqkm'].max()
    if max_drainage_density > 0:
        wards_gdf['drainage_risk_factor'] = (max_drainage_density - wards_gdf['drainage_density_km_sqkm']) / max_drainage_density
    else:
        wards_gdf['drainage_risk_factor'] = 0
    
    return wards_gdf

@st.cache_data(ttl=3600)
def calculate_composite_resilience_index(_wards_gdf, _rainfall_df) -> gpd.GeoDataFrame:
    """
    Calculates a comprehensive, multi-factor resilience index for each ward.
    
    Args:
        _wards_gdf (gpd.GeoDataFrame): GeoDataFrame of BBMP wards with incident/drainage metrics.
        _rainfall_df (pd.DataFrame): DataFrame of historical rainfall data.
        
    Returns:
        gpd.GeoDataFrame: Wards GeoDataFrame with the new resilience index and risk levels.
    """
    wards_gdf = _wards_gdf.copy()
    
    # Define weighting for the composite score (these are heuristic values for demonstration)
    WEIGHT_INCIDENT_DENSITY = 0.4
    WEIGHT_PROXIMITY_INCIDENTS = 0.2
    WEIGHT_DRAINAGE_EFFICIENCY = 0.4 # Inverse, so less drain is higher risk
    
    # Normalize metrics to a 0-1 scale to allow for fair weighting
    # Incident Density
    max_incident_density = wards_gdf['incident_density_sqkm'].max()
    if max_incident_density > 0:
        wards_gdf['normalized_incident_density'] = wards_gdf['incident_density_sqkm'] / max_incident_density
    else:
        wards_gdf['normalized_incident_density'] = 0
        
    # Proximity Incidents (log transform to prevent a few outliers from dominating)
    wards_gdf['log_buffered_incidents'] = wards_gdf['buffered_incident_count'].apply(lambda x: math.log1p(x))
    max_log_buffered_incidents = wards_gdf['log_buffered_incidents'].max()
    if max_log_buffered_incidents > 0:
        wards_gdf['normalized_proximity'] = wards_gdf['log_buffered_incidents'] / max_log_buffered_incidents
    else:
        wards_gdf['normalized_proximity'] = 0

    # Drainage Efficiency (inverse relationship, already calculated as `drainage_risk_factor`)
    wards_gdf['normalized_drainage_risk'] = wards_gdf['drainage_risk_factor']
    
    # Calculate the Composite Resilience Index
    wards_gdf['Composite_Resilience_Index'] = (
        (wards_gdf['normalized_incident_density'] * WEIGHT_INCIDENT_DENSITY) +
        (wards_gdf['normalized_proximity'] * WEIGHT_PROXIMITY_INCIDENTS) +
        (wards_gdf['normalized_drainage_risk'] * WEIGHT_DRAINAGE_EFFICIENCY)
    )

    # Normalize the final index to a scale of 0-100 for easier interpretation
    max_index = wards_gdf['Composite_Resilience_Index'].max()
    if max_index > 0:
        wards_gdf['Composite_Resilience_Index'] = (wards_gdf['Composite_Resilience_Index'] / max_index) * 100
    else:
        wards_gdf['Composite_Resilience_Index'] = 0
        
    # Assign Risk Level based on the new index
    def assign_resilience_level(score):
        if score >= 85: return "Extreme Vulnerability"
        elif score >= 60: return "High Vulnerability"
        elif score >= 35: return "Moderate Vulnerability"
        elif score >= 10: return "Low Vulnerability"
        else: return "High Resilience"

    wards_gdf['resilience_level'] = wards_gdf['Composite_Resilience_Index'].apply(assign_resilience_level)
    
    return wards_gdf

# ==============================================================================
# DATA LOADING & INITIALIZATION
# ==============================================================================

# Load all data at startup
bbmp_wards_raw, primary_drains, all_flood_points_gdf = load_geospatial_data()
rainfall_data = load_tabular_data()

# Calculate and integrate all metrics
bbmp_wards_metrics = calculate_flood_incident_metrics(bbmp_wards_raw, all_flood_points_gdf)
bbmp_wards_metrics = calculate_drainage_metrics(bbmp_wards_metrics, primary_drains)
bbmp_wards = calculate_composite_resilience_index(bbmp_wards_metrics, rainfall_data)

# Check if data loading was successful
if bbmp_wards is None:
    st.error("FATAL ERROR: Data initialization failed.")
    st.stop()


# ==============================================================================
# STYLING & COLOR PALETTES
# ==============================================================================

# New color palettes for the Composite Resilience Index
resilience_colors = {
    "Extreme Vulnerability": "#8B0000",      # Dark Red
    "High Vulnerability": "#FF4500",         # OrangeRed
    "Moderate Vulnerability": "#FFD700",    # Gold
    "Low Vulnerability": "#32CD32",          # LimeGreen
    "High Resilience": "#008000"             # Green
}

# Colors for the new "What-If" simulation
simulated_colors = {
    "Catastrophic": "#8B0000",       # Dark Red
    "Severe Flooding": "#DC143C",     # Crimson
    "Significant Flooding": "#FF4500",  # OrangeRed
    "Minor Flooding": "#FFD700",       # Gold
    "Low Impact": "#32CD32"            # LimeGreen
}

# Grid Risk Colors - designed for clear visibility and distinction
grid_risk_colors = {
    "Critical Risk": "#8B0000",
    "High Risk": "#B22222",
    "Moderate Risk": "#FF8C00",
    "Low Risk": "#3CB371",
    "Minor Risk": "#6B8E23",
    "No Incidents": "#00000000"
}

def assign_grid_risk_level(incident_count):
    """Assigns a risk level to a grid cell based on incident count."""
    if incident_count == 0: return "No Incidents"
    elif incident_count == 1: return "Minor Risk"
    elif incident_count <= 3: return "Low Risk"
    elif incident_count <= 6: return "Moderate Risk"
    elif incident_count <= 10: return "High Risk"
    else: return "Critical Risk"

# Custom CSS for a professional, high-impact dark look
st.set_page_config(
    page_title="💧 HAURCC - Bengaluru (Urban Resilience Command Center)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base64 SVG for a more dynamic and integrated icon
svg_icon_b64 = base64.b64encode(b"""
<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='#E0E0E0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' class='feather feather-layers'><polygon points='12 2 2 7 12 12 22 7 12 2'></polygon><polyline points='2 17 12 22 22 17'></polyline><polyline points='2 12 12 17 22 12'></polyline></svg>
""").decode()

st.markdown(
    f"""
    <style>
    /* Google Fonts for a modern, crisp look */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&family=Montserrat:wght@300;400;600;700&display=swap');

    html, body, [class*="st-emotion-cache"] {{
        font-family: 'Montserrat', sans-serif;
        color: #E0E0E0;
    }}
    .stApp {{ background-color: #0F0F0F; }}
    h1 {{
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        font-size: 3.5em;
        background: linear-gradient(45deg, #00C0FF, #00FF99, #9966FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 10px rgba(0,255,255,0.3), 2px 2px 15px rgba(0,0,0,0.5);
    }}
    h2, h3, h4, h5, h6 {{
        font-family: 'Roboto Mono', monospace;
        color: #00FF99;
        font-weight: 600;
        margin-top: 1.8em;
        margin-bottom: 0.8em;
    }}
    h2.map-heading {{
        color: #00C0FF;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(0,192,255,0.4);
    }}
    .st-emotion-cache-nahz7x div, .st-emotion-cache-1eereed, .stPlotlyChart, .st-emotion-cache-1cpxdgc, .st-emotion-cache-nahz7x {{
        background-color: #1A1A1A;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 40px rgba(0,255,255,0.1), 0 5px 15px rgba(0,0,0,0.4);
        margin-bottom: 30px;
        border: 1px solid #2C3E50;
        transition: transform 0.2s ease-in-out, box-shadow 0.3s ease;
    }}
    .st-emotion-cache-nahz7x:hover, .st-emotion-cache-1eereed:hover, .stPlotlyChart:hover, .st-emotion-cache-1cpxdgc:hover {{
        transform: translateY(-7px);
        box-shadow: 0 15px 50px rgba(0,255,255,0.2), 0 8px 20px rgba(0,0,0,0.6);
    }}
    .stSelectbox, .stTextInput, .stButton, .stSlider {{
        border-radius: 10px;
        border: 1px solid #34495E;
        padding: 8px;
        background-color: #2C3E50;
        color: #E0E0E0;
    }}
    .stSelectbox div[data-baseweb="select"] > div, .stTextInput div[data-baseweb="input"] > input {{
        color: #E0E0E0 !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div:hover {{
        background-color: #34495E;
    }}
    .stButton>button {{
        background-color: #00FF99;
        color: #0A0A0A;
        border-radius: 10px;
        padding: 12px 25px;
        font-weight: 700;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 6px 15px rgba(0, 255, 153, 0.4);
        letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{
        background-color: #00E088;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 255, 153, 0.6), 0 0 25px rgba(0, 255, 153, 0.4);
    }}
    .stAlert {{
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.4);
        color: #E0E0E0;
    }}
    .stAlert.info {{ background-color: #2C3E50; border-left: 6px solid #00C0FF; }}
    .stAlert.warning {{ background-color: #8C5D00; border-left: 6px solid #FFD700; }}
    .stAlert.success {{ background-color: #1F6A3F; border-left: 6px solid #00FF99; }}
    .stAlert.error {{ background-color: #7B241C; border-left: 6px solid #E61A00; }}
    .css-1d391kg {{
        width: 400px;
        background-color: #1A1A1A;
        box-shadow: 5px 0 20px rgba(0,0,0,0.3);
        padding-top: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    .st-emotion-cache-nahz7x p, .st-emotion-cache-nahz7x ul, .st-emotion-cache-nahz7x li, .st-emotion-cache-1cpxdgc p {{
        line-height: 1.7;
        color: #E0E0E0;
    }}
    .st-emotion-cache-nahz7x b, .st-emotion-cache-1cpxdgc b {{
        color: #00C0FF;
        font-weight: 700;
    }}
    .st-emotion-cache-nahz7x ul li, .st-emotion-cache-1cpxdgc ul li {{
        margin-bottom: 0.6em;
    }}
    .st-emotion-cache-nahz7x span, .st-emotion-cache-1cpxdgc span {{
        color: #E0E0E0;
    }}
    .st-emotion-cache-nahz7x strong, .st-emotion-cache-1cpxdgc strong {{
        color: #E0E0E0;
    }}
    .leaflet-control.leaflet-control-custom.leaflet-control-layers-expanded {{
        padding: 12px;
        background-color: #1A1A1A;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif;
    }}
    .leaflet-control-layers-toggle {{
        background-image: url('data:image/svg+xml;base64,{svg_icon_b64}') !important;
        background-size: 20px 20px;
        background-repeat: no-repeat;
        background-position: center center;
    }}
    .legend-i {{
        width: 20px;
        height: 20px;
        float: left;
        margin-right: 10px;
        opacity: 0.9;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.2);
    }}
    .leaflet-tooltip {{
        background-color: #2C3E50 !important;
        color: #E0E0E0 !important;
        border: 1px solid #34495E !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.4);
        font-family: 'Montserrat', sans-serif !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
    }}
    .leaflet-popup-content-wrapper {{
        background-color: #2C3E50 !important;
        color: #E0E0E0 !important;
        border-radius: 10px !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.4);
        font-family: 'Montserrat', sans-serif !important;
        font-size: 14px !important;
    }}
    .leaflet-popup-tip {{ background: #2C3E50 !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# STREAMLIT UI - DASHBOARD LAYOUT
# ==============================================================================

# Header Section
st.container()
st.markdown("<h1>💧 HAURCC - Bengaluru</h1>", unsafe_allow_html=True)
st.markdown(f"""
    <p style='font-size: 1.2em; color: #BDC3C7; font-weight: 400;'>
        **Hyper-Analytical Urban Resilience Command Center** - Empowering proactive planning for a flood-resilient Bengaluru.
        This system provides multi-dimensional risk assessment, dynamic geospatial insights, and strategic analysis
        derived from robust civic and historical data for {CURRENT_MONTH_YEAR}.
    </p>
    """, unsafe_allow_html=True)
st.markdown("---")


# Main Content Area
col1, col2 = st.columns([0.7, 0.3])

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00FF99;'>🗺️ Command Center Controls</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.05em; text-align: center; color: #BDC3C7;'>Navigate and configure the HAURCC display modes and insights.</p>", unsafe_allow_html=True)

    # --- Ward Selection & Display ---
    ward_names = sorted(bbmp_wards['KGISWardName'].dropna().unique().tolist())
    ward_options = ["--- Bangalore City Overview ---"] + ward_names

    selected_ward_name = st.selectbox(
        "**🎯 Select Target Ward:**",
        options=ward_options,
        key="ward_selector",
        help="Choose 'Bangalore City Overview' for a macro view, or a specific ward for granular analysis."
    )

    selected_ward_gdf = None
    if selected_ward_name != " Bangalore City ":
        selected_ward_gdf = bbmp_wards[bbmp_wards['KGISWardName'] == selected_ward_name].copy()
        if not selected_ward_gdf.empty:
            display_properties = selected_ward_gdf.iloc[0].to_dict()
            
            st.markdown(f"<h3 style='color: #00C0FF;'>🏡 Ward: {display_properties.get('KGISWardName', 'N/A')}</h3>", unsafe_allow_html=True)
            
            # Display core metrics for the selected ward
            resilience_level = display_properties.get('resilience_level', 'High Resilience')
            resilience_score = display_properties.get('Composite_Resilience_Index', 0)
            
            st.markdown(f"**Ward No.:** <span style='font-size: 1.1em; color: #E0E0E0;'>{display_properties.get('KGISWardNo', 'N/A')}</span>", unsafe_allow_html=True)
            st.markdown(f"**Area:** <span style='font-size: 1.1em; color: #E0E0E0;'>{display_properties.get('area_sqkm', 0):.2f} km²</span>", unsafe_allow_html=True)
            st.markdown(f"**Calculated Resilience:** <span style='color: {resilience_colors.get(resilience_level)}; font-weight: bold; font-size: 1.1em;'>{resilience_level}</span>", unsafe_allow_html=True)
            st.markdown(f"**Resilience Index:** <span style='font-weight: bold; color: #00FF99;'>{resilience_score:.2f} / 100</span>", unsafe_allow_html=True)
            
            # Granular Hotspot Analysis Controls
            st.markdown("---")
            st.markdown("<h3 class='map-heading'>⚙️ Granular Hotspot Analysis</h3>", unsafe_allow_html=True)
            st.markdown("""
                <p style='font-size: 0.9em; font-style: italic; color: #BDC3C7;'>
                Configure the resolution of the localized grid to pinpoint high-risk micro-zones within the ward.
                Hover over cells for incident counts.
                </p>
                """, unsafe_allow_html=True)
            
            grid_size_m_option = st.slider(
                "**Grid Cell Size (meters):**",
                min_value=100, max_value=500, value=250, step=50,
                help="Adjust the grid resolution for detailed hotspot analysis within the selected ward."
            )
            st.session_state['grid_size_m'] = grid_size_m_option
            
            st.markdown("---")

            # "What-If" Simulation Controls
            st.markdown("<h3 class='map-heading'>🔬 What-If Simulation</h3>", unsafe_allow_html=True)
            st.markdown("""
                <p style='font-size: 0.9em; font-style: italic; color: #BDC3C7;'>
                Simulate a hypothetical rainfall event to see the potential impact on flood risk.
                A multiplier of 1.0 represents average monthly rainfall.
                </p>
                """, unsafe_allow_html=True)
            
            rainfall_multiplier = st.slider(
                "**Rainfall Multiplier:**",
                min_value=0.5, max_value=5.0, value=1.0, step=0.1,
                help="Select a value to represent a rainfall event (e.g., 2.0 for double the normal rainfall)."
            )
            
            if st.button("Run Simulation"):
                st.session_state['simulation_run'] = True
                st.session_state['rainfall_multiplier'] = rainfall_multiplier
            else:
                st.session_state['simulation_run'] = False
        
        else:
            st.info("No detailed data found for the selected ward. Please choose another ward.", icon="ℹ️")
            
    else:
        st.info("Select a specific ward from the dropdown above to unlock detailed insights and granular hotspot analysis.", icon="💡")


# ==============================================================================
# MAP GENERATION & DISPLAY
# ==============================================================================

with col1:
    map_center = [12.9716, 77.5946]
    zoom_level = 11

    basemaps = {
        "OpenStreetMap": folium.TileLayer(tiles="OpenStreetMap", name="Base Map: OpenStreetMap", attr="OpenStreetMap contributors"),
        "CartoDB Positron": folium.TileLayer(tiles="CartoDB Positron", name="Base Map: Positron", attr="CartoDB, OpenStreetMap contributors"),
        "Stamen TonerLite": folium.TileLayer(tiles="Stamen TonerLite", name="Base Map: Toner Lite", attr="Stamen Design, OpenStreetMap contributors"),
    }
    
    m = folium.Map(location=map_center, zoom_start=zoom_level, control_scale=True, tiles=basemaps["CartoDB Positron"])
    for name, tile_layer_obj in basemaps.items():
        if name != "CartoDB Positron":
            tile_layer_obj.add_to(m)

    if selected_ward_name == "--- Bangalore City Overview ---":
        st.markdown("<h2 class='map-heading'>🏙️ Bengaluru City-Wide Flood Resilience Overview</h2>", unsafe_allow_html=True)
        st.info("Visualizing city-wide resilience index. Zoom and pan to explore. Use layer controls to toggle different data views.", icon="🗺️")

        # Add all BBMP Wards, colored by their new resilience level
        folium.GeoJson(
            bbmp_wards.__geo_interface__,
            name="HAURCC: Ward Resilience Index",
            style_function=lambda feature: {
                "fillColor": resilience_colors.get(feature['properties'].get('resilience_level', 'High Resilience')),
                "color": "#333333",
                "weight": 0.8,
                "fillOpacity": 0.75
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['KGISWardName', 'KGISWardNo', 'Composite_Resilience_Index', 'resilience_level'],
                aliases=['Ward Name:', 'Ward No.:', 'Resilience Index:', 'Resilience Level:'],
                localize=True,
                style="background-color: #2C3E50; color: #E0E0E0; font-family: 'Montserrat', sans-serif; font-size: 14px; border: 1px solid #34495E; padding: 10px;"
            )
        ).add_to(m)

        # Add a custom legend for the resilience levels
        legend_html = f"""
             <div style="position: fixed;
                         bottom: 50px; left: 50px; width: 220px; height: 230px;
                         border:2px solid #2C3E50; z-index:9999; font-size:14px;
                         background-color:#1A1A1A; opacity:0.95; padding:15px; border-radius:12px;
                         box-shadow: 0 5px 15px rgba(0,0,0,0.4); color: #E0E0E0; font-family: 'Montserrat', sans-serif;">
               &nbsp; <b>Ward Resilience Index</b> <br>
               &nbsp; <i class="legend-i" style="background:{resilience_colors['Extreme Vulnerability']}"></i> Extreme Vulnerability <br>
               &nbsp; <i class="legend-i" style="background:{resilience_colors['High Vulnerability']}"></i> High Vulnerability <br>
               &nbsp; <i class="legend-i" style="background:{resilience_colors['Moderate Vulnerability']}"></i> Moderate Vulnerability <br>
               &nbsp; <i class="legend-i" style="background:{resilience_colors['Low Vulnerability']}"></i> Low Vulnerability <br>
               &nbsp; <i class="legend-i" style="background:{resilience_colors['High Resilience']}"></i> High Resilience <br>
             </div>
             """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add City-Wide Incident Density Heatmap
        coords = [[p.y, p.x] for p in all_flood_points_gdf.geometry if p and p.geom_type == 'Point']
        if coords:
            folium.plugins.HeatMap(coords, name="Global Flood Incident Density", radius=15, blur=10, max_zoom=14).add_to(m)

    else: # A specific ward is selected
        if selected_ward_gdf is not None and not selected_ward_gdf.empty:
            st.markdown(f"<h2 class='map-heading'>📍 HAURCC: {selected_ward_name} Detailed Analysis</h2>", unsafe_allow_html=True)
            
            # Center map on the selected ward
            map_center = [selected_ward_gdf.geometry.centroid.y.iloc[0], selected_ward_gdf.geometry.centroid.x.iloc[0]]
            zoom_level = 14
            m.location = map_center
            m.zoom_start = zoom_level

            # Check if a simulation is running
            if st.session_state.get('simulation_run', False) and 'simulated_risk_level' in bbmp_wards.columns:
                simulated_gdf = simulate_rainfall_impact(bbmp_wards, st.session_state['rainfall_multiplier'])
                sim_ward_gdf = simulated_gdf[simulated_gdf['KGISWardName'] == selected_ward_name].iloc[0]
                
                st.warning(f"**Simulation Active:** Viewing hypothetical flood risk for a **{st.session_state['rainfall_multiplier']}x** rainfall event.", icon="⚠️")
                st.markdown(f"<p style='font-size: 1.1em; color: #FFD700;'>Simulated Risk Level: <b style='color: {simulated_colors.get(sim_ward_gdf['simulated_risk_level'])}'>{sim_ward_gdf['simulated_risk_level']}</b></p>", unsafe_allow_html=True)
                
                folium.GeoJson(
                    simulated_gdf.__geo_interface__,
                    name=f"Simulated Flood Risk ({st.session_state['rainfall_multiplier']}x Rainfall)",
                    style_function=lambda feature: {
                        "fillColor": simulated_colors.get(feature['properties'].get('simulated_risk_level', 'Low Impact')),
                        "color": "#333333",
                        "weight": 0.5,
                        "fillOpacity": 0.85
                    },
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['KGISWardName', 'simulated_impact_score', 'simulated_risk_level'],
                        aliases=['Ward:', 'Simulated Score:', 'Simulated Risk:'],
                        localize=True,
                        style="background-color: #2C3E50; color: #E0E0E0; font-family: 'Montserrat', sans-serif; font-size: 14px; border: 1px solid #34495E; padding: 10px;"
                    )
                ).add_to(m)

                # Add a legend for the simulation
                sim_legend_html = f"""
                     <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: 210px; border:2px solid #2C3E50; z-index:9999; font-size:14px; background-color:#1A1A1A; opacity:0.95; padding:15px; border-radius:12px; box-shadow: 0 5px 15px rgba(0,0,0,0.4); color: #E0E0E0; font-family: 'Montserrat', sans-serif;">
                       &nbsp; <b>Simulated Flood Risk</b> <br>
                       &nbsp; <i class="legend-i" style="background:{simulated_colors['Catastrophic']}"></i> Catastrophic <br>
                       &nbsp; <i class="legend-i" style="background:{simulated_colors['Severe Flooding']}"></i> Severe Flooding <br>
                       &nbsp; <i class="legend-i" style="background:{simulated_colors['Significant Flooding']}"></i> Significant Flooding <br>
                       &nbsp; <i class="legend-i" style="background:{simulated_colors['Minor Flooding']}"></i> Minor Flooding <br>
                       &nbsp; <i class="legend-i" style="background:{simulated_colors['Low Impact']}"></i> Low Impact <br>
                     </div>
                     """
                m.get_root().html.add_child(folium.Element(sim_legend_html))
                
            else:
                # Add the SELECTED BBMP Ward boundary with its resilience color
                folium.GeoJson(
                    selected_ward_gdf.__geo_interface__,
                    name=f"Selected Ward: {selected_ward_name}",
                    style_function=lambda feature: {
                        "fillColor": resilience_colors.get(feature['properties'].get('resilience_level', 'High Resilience')),
                        "color": "#000000",
                        "weight": 3.5,
                        "fillOpacity": 0.45
                    },
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['KGISWardName', 'KGISWardNo', 'Composite_Resilience_Index', 'resilience_level'],
                        aliases=['Ward Name:', 'Ward No.:', 'Resilience Index:', 'Resilience Level:'],
                        localize=True,
                        style="background-color: #2C3E50; color: #E0E0E0; font-family: 'Montserrat', sans-serif; font-size: 14px; border: 1px solid #34495E; padding: 10px;"
                    )
                ).add_to(m)

            # Grid Generation and Display for Selected Ward
            try:
                clicked_shape = selected_ward_gdf.geometry.iloc[0]
                clicked_gdf = gpd.GeoDataFrame([1], geometry=[clicked_shape], crs="EPSG:4326")
                clicked_gdf_proj = clicked_gdf.to_crs("EPSG:32643")
                
                minx, miny, maxx, maxy = clicked_gdf_proj.total_bounds
                grid_size_meters = st.session_state.get('grid_size_m', 250)
                polygons = []
                x_coords = np.arange(minx, maxx + grid_size_meters, grid_size_meters)
                y_coords = np.arange(miny, maxy + grid_size_meters, grid_size_meters)

                for i in range(len(x_coords) - 1):
                    for j in range(len(y_coords) - 1):
                        grid_cell_proj = box(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
                        if clicked_gdf_proj.geometry.iloc[0].intersects(grid_cell_proj):
                            polygons.append(grid_cell_proj)
                
                if polygons:
                    grid_gdf_proj = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:32643")
                    grid_gdf = grid_gdf_proj.to_crs("EPSG:4326")

                    ward_bounds = selected_ward_gdf.total_bounds
                    bbox_poly = box(ward_bounds[0], ward_bounds[1], ward_bounds[2], ward_bounds[3])
                    relevant_flood_points = all_flood_points_gdf[all_flood_points_gdf.geometry.intersects(bbox_poly)]

                    grid_with_points = gpd.sjoin(grid_gdf, relevant_flood_points, how="left", predicate="intersects")
                    incident_counts_per_grid_cell = grid_with_points.groupby(grid_with_points.index).size().rename("incident_count_in_cell")
                    
                    grid_gdf = grid_gdf.merge(incident_counts_per_grid_cell, left_index=True, right_index=True, how="left")
                    grid_gdf['incident_count_in_cell'] = grid_gdf['incident_count_in_cell'].fillna(0).astype(int)
                    grid_gdf['grid_risk_level'] = grid_gdf['incident_count_in_cell'].apply(assign_grid_risk_level)
                    
                    folium.GeoJson(
                        grid_gdf.__geo_interface__,
                        name=f"{st.session_state['grid_size_m']}m Grid Hotspots",
                        style_function=lambda feature: {
                            "color": "#A0A0A0",
                            "weight": 0.7,
                            "fillColor": grid_risk_colors.get(feature['properties'].get('grid_risk_level', 'No Incidents')),
                            "fillOpacity": 0.8 if feature['properties'].get('incident_count_in_cell', 0) > 0 else 0.0,
                        },
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=['incident_count_in_cell', 'grid_risk_level'],
                            aliases=['Flood Incidents in this cell:', 'Grid Risk Level:'],
                            localize=True,
                            style="background-color: #2C3E50; color: #E0E0E0; font-family: 'Montserrat', sans-serif; font-size: 14px; border: 1px solid #34495E; padding: 10px;"
                        )
                    ).add_to(m)
                    st.info(f"Viewing localized flood hotspots with a {st.session_state['grid_size_m']}m grid. Hover over colored cells for incident counts.", icon="🔎")

                    grid_legend_html = f"""
                             <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 230px; border:2px solid #2C3E50; z-index:9999; font-size:14px; background-color:#1A1A1A; opacity:0.95; padding:15px; border-radius:12px; box-shadow: 0 5px 15px rgba(0,0,0,0.4); color: #E0E0E0; font-family: 'Montserrat', sans-serif;">
                                 &nbsp; <b>Grid Hotspot Risk</b> <br>
                                 &nbsp; <i class="legend-i" style="background:{grid_risk_colors['Critical Risk']}"></i> Critical Risk <br>
                                 &nbsp; <i class="legend-i" style="background:{grid_risk_colors['High Risk']}"></i> High Risk <br>
                                 &nbsp; <i class="legend-i" style="background:{grid_risk_colors['Moderate Risk']}"></i> Moderate Risk <br>
                                 &nbsp; <i class="legend-i" style="background:{grid_risk_colors['Low Risk']}"></i> Low Risk <br>
                                 &nbsp; <i class="legend-i" style="background:{grid_risk_colors['Minor Risk']}"></i> Minor Risk <br>
                                 &nbsp; <i class="legend-i" style="background:{grid_risk_colors['No Incidents']}"></i> No Incidents <br>
                             </div>
                             """
                    m.get_root().html.add_child(folium.Element(grid_legend_html))
                else:
                    st.warning(f"Could not generate any intersecting {st.session_state.get('grid_size_m', 250)}m grid cells for {selected_ward_name}.", icon="⚠️")

            except Exception as e:
                st.error(f"Error during grid generation for {selected_ward_name}: {e}", icon="❌")
                st.exception(e)

        else:
            st.warning(f"No GeoData found for ward: {selected_ward_name}. Please check the ward name in your GeoJSON data.", icon="⚠️")

    # Add Primary Stormwater Drains Layer
    if not primary_drains.empty:
        folium.GeoJson(
            primary_drains.__geo_interface__,
            name="Primary Stormwater Drains",
            style_function=lambda x: {"color": "#0099FF", "weight": 2.5, "opacity": 0.8},
            tooltip=folium.features.GeoJsonTooltip(
                fields=['Name', 'Description', 'length_km'],
                aliases=['Drain Name:', 'Description:', 'Length (km):'],
                localize=True,
                style="background-color: #2C3E50; color: #E0E0E0; font-family: 'Montserrat', sans-serif; font-size: 14px; border: 1px solid #34495E; padding: 10px;"
            )
        ).add_to(m)

    # Add ALL Flood Incident Points (Historical Markers)
    if not all_flood_points_gdf.empty:
        mc = folium.plugins.MarkerCluster(name="Historical Flood Incidents (Clusters)").add_to(m)
        for idx, row in all_flood_points_gdf.iterrows():
            if row.geometry and row.geometry.geom_type == 'Point':
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color='#CC0000',
                    fill=True,
                    fill_color='#FF0000',
                    fill_opacity=0.9,
                    tooltip=folium.Tooltip(
                        f"<b>Incident:</b> {row.get('Name', 'N/A')}<br>"
                        f"<b>Location:</b> {row.get('LocationName', 'N/A')}<br>"
                        f"<b>Ward:</b> {row.get('WARD_NAME', 'N/A')}<br>"
                        f"<b>Ward No.:</b> {row.get('WARDNO', 'N/A')}",
                        style="font-family: 'Montserrat', sans-serif; font-size: 14px; background-color: #2C3E50; color: #E0E0E0; border: 1px solid #34495E; padding: 10px;"
                    )
                ).add_to(mc)

    # Add Layer Control to toggle layers
    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, width='100%', height=650, key="haurcc_map_display", returned_objects=[])

with col2:
    st.markdown("<p style='font-size: 1.1em; color: #BDC3C7;'>The interactive map provides the core operational view:</p>", unsafe_allow_html=True)
    st.markdown("""
        <ul style='font-size: 1em; color: #E0E0E0;'>
            <li><b>Resilience Index:</b> Wards are colored by their comprehensive, multi-factor resilience index.</li>
            <li><b>Dynamic Grid Hotspots:</b> (Ward View) High-resolution grid cells illuminate micro-level incident concentrations.</li>
            <li><b>Simulation Layers:</b> (Ward View) See the hypothetical impact of a heavy rainfall event.</li>
            <li><b>Primary Drains Network:</b> Toggle to visualize critical stormwater drainage infrastructure.</li>
            <li><b>Historical Incidents:</b> Explore individual past flood event locations.</li>
            <li><b>Customizable Basemaps:</b> Switch background maps for optimal data contrast.</li>
        </ul>
        <p style='font-size: 1em; color: #BDC3C7;'>
            Use the layer control icon in the top right of the map to toggle layers.
        </p>
    """, unsafe_allow_html=True)


# ==============================================================================
# ADVANCED ANALYTICS SECTION
# ==============================================================================

st.markdown("---")
st.container()
st.markdown("<h2 style='color: #00FF99;'>📈 Advanced Resilience Analytics</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1em; color: #BDC3C7;'>Dive deeper into Bengaluru's urban resilience profile with advanced data visualizations and comparative analysis.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Rainfall Patterns", "🤝 Ward Comparison", "📋 Incident Breakdown", "📈 Resilience Index Distribution"])


with tab1: # Rainfall Patterns
    st.markdown("<h3 style='color: #00C0FF;'>🌧️ Historical Rainfall Trends & Anomalies</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95em; color: #E0E0E0;'>Analyze annual and monthly rainfall patterns, including deviations from the long-term mean.</p>", unsafe_allow_html=True)

    if rainfall_data is not None and not rainfall_data.empty:
        # Annual Rainfall Chart
        annual_rainfall_chart = alt.Chart(rainfall_data).mark_line(point=True, color='#00C0FF').encode(
            x=alt.X('Year:O', axis=alt.Axis(format='d', title='Year', titleColor='#E0E0E0', labelColor='#E0E0E0', labelAngle=-45)),
            y=alt.Y('Total:Q', axis=alt.Axis(title='Total Annual Rainfall (mm)', titleColor='#E0E0E0', labelColor='#E0E0E0')),
            tooltip=[alt.Tooltip('Year'), alt.Tooltip('Total', format='.1f', title='Rainfall')]
        ).properties(
            title=alt.Title('Total Annual Rainfall (1901-2024)', anchor='start', fontSize=18, color='#E0E0E0')
        ).configure_axis(
            gridColor='#34495E', domainColor='#34495E', tickColor='#34495E',
        ).configure_view(
            strokeWidth=0,
            fill='#1A1A1A'
        ).interactive()
        st.altair_chart(annual_rainfall_chart, use_container_width=True)

        # Annual Rainfall Deviation Chart
        deviation_chart = alt.Chart(rainfall_data).mark_bar().encode(
            x=alt.X('Year:O', axis=alt.Axis(format='d', title='Year', titleColor='#E0E0E0', labelColor='#E0E0E0', labelAngle=-45)),
            y=alt.Y('deviation_from_mean:Q', title='Deviation from Mean Annual Rainfall (mm)', axis=alt.Axis(titleColor='#E0E0E0', labelColor='#E0E0E0')),
            color=alt.condition(
                alt.datum.deviation_from_mean > 0,
                alt.value('#00FF99'),
                alt.value('#FF4500')
            ),
            tooltip=[
                alt.Tooltip('Year'),
                alt.Tooltip('Total', format='.1f', title='Actual Rainfall'),
                alt.Tooltip('deviation_from_mean', format='.1f', title='Deviation')
            ]
        ).properties(
            title=alt.Title('Annual Rainfall Deviation from Long-term Average', anchor='start', fontSize=18, color='#E0E0E0')
        ).configure_axis(
            gridColor='#34495E', domainColor='#34495E', tickColor='#34495E',
        ).configure_view(
            strokeWidth=0,
            fill='#1A1A1A'
        ).interactive()
        st.altair_chart(deviation_chart, use_container_width=True)


with tab2: # Ward Comparison
    st.markdown("<h3 style='color: #00C0FF;'>🤝 Ward Performance Comparison</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95em; color: #E0E0E0;'>Compare key resilience metrics across multiple BBMP wards side-by-side to identify best practices and areas needing urgent intervention.</p>", unsafe_allow_html=True)

    wards_for_comparison = st.multiselect(
        "**Select Wards for Comparison:**",
        options=ward_names,
        default=ward_names[:3],
        key="comparison_wards_selector",
        help="Choose 2-5 wards to compare their resilience metrics."
    )

    if len(wards_for_comparison) > 0:
        comparison_gdf = bbmp_wards[bbmp_wards['KGISWardName'].isin(wards_for_comparison)].copy()
        if not comparison_gdf.empty:
            # Prepare data for Altair chart
            comparison_metrics = comparison_gdf[[
                'KGISWardName',
                'incident_density_sqkm',
                'drainage_density_km_sqkm',
                'Composite_Resilience_Index',
                'normalized_proximity'
            ]].set_index('KGISWardName').T.reset_index()
            comparison_metrics = comparison_metrics.rename(columns={'index': 'Metric'})

            melted_comparison = comparison_metrics.melt(
                id_vars=['Metric'], var_name='Ward', value_name='Value'
            )

            comparison_chart = alt.Chart(melted_comparison).mark_bar().encode(
                x=alt.X('Ward:N', title='Ward Name', axis=alt.Axis(titleColor='#E0E0E0', labelColor='#E0E0E0', labelAngle=-45)),
                y=alt.Y('Value:Q', title='Metric Value (Normalized)', axis=alt.Axis(titleColor='#E0E0E0', labelColor='#E0E0E0')),
                color=alt.Color('Ward:N', legend=alt.Legend(title="Ward", titleColor='#E0E0E0', labelColor='#E0E0E0')),
                column=alt.Column('Metric:N', header=alt.Header(titleOrient="bottom", labelOrient="bottom", titleColor='#00FF99', labelColor='#E0E0E0')),
                tooltip=[alt.Tooltip('Ward'), alt.Tooltip('Metric'), alt.Tooltip('Value', format='.2f')]
            ).properties(
                title=alt.Title('Ward Resilience Metric Comparison', anchor='start', fontSize=18, color='#E0E0E0')
            ).configure_axis(
                gridColor='#34495E', domainColor='#34495E', tickColor='#34495E',
            ).configure_view(
                strokeWidth=0, fill='#1A1A1A'
            ).interactive()
            st.altair_chart(comparison_chart, use_container_width=True)
        else:
            st.warning("No data available for selected wards. Please check your selection.", icon="⚠️")
    else:
        st.info("Select at least one ward for comparison from the dropdown above.", icon="💡")


with tab3: # Incident Breakdown
    st.markdown("<h3 style='color: #00C0FF;'>⚠️ Incident Breakdown by Ward</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95em; color: #E0E0E0;'>Analyze the distribution of historical flood incidents across different wards.</p>", unsafe_allow_html=True)

    if not all_flood_points_gdf.empty and 'WARD_NAME' in all_flood_points_gdf.columns:
        incident_ward_counts = all_flood_points_gdf['WARD_NAME'].value_counts().reset_index()
        incident_ward_counts.columns = ['Ward', 'Count']

        ward_breakdown_chart = alt.Chart(incident_ward_counts).mark_bar().encode(
            x=alt.X('Count:Q', title='Number of Incidents', axis=alt.Axis(titleColor='#E0E0E0', labelColor='#E0E0E0')),
            y=alt.Y('Ward:N', sort='-x', title='Ward Name', axis=alt.Axis(titleColor='#E0E0E0', labelColor='#E0E0E0')),
            color=alt.Color('Ward:N', legend=None),
            tooltip=['Ward', 'Count']
        ).properties(
            title=alt.Title('Historical Flood Incident Breakdown by Ward', anchor='start', fontSize=18, color='#E0E0E0')
        ).configure_axis(
            gridColor='#34495E', domainColor='#34495E', tickColor='#34495E',
        ).configure_view(
            strokeWidth=0, fill='#1A1A1A'
        ).interactive()

        st.altair_chart(ward_breakdown_chart, use_container_width=True)
    else:
        st.warning("Ward name data for incident breakdown is not available. Please ensure the 'WARD_NAME' column exists in your flood incident data.", icon="⚠️")
        

with tab4: # Resilience Index Distribution
    st.markdown("<h3 style='color: #00C0FF;'>📊 Resilience Index Distribution</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95em; color: #E0E0E0;'>Visualize the distribution of the Composite Resilience Index across all BBMP wards to identify the overall resilience profile of the city.</p>", unsafe_allow_html=True)
    
    if not bbmp_wards.empty and 'Composite_Resilience_Index' in bbmp_wards.columns:
        distribution_chart = alt.Chart(bbmp_wards).mark_bar().encode(
            x=alt.X('Composite_Resilience_Index:Q', bin=True, title='Resilience Index Score (0-100)', axis=alt.Axis(titleColor='#E0E0E0', labelColor='#E0E0E0')),
            y=alt.Y('count():Q', title='Number of Wards', axis=alt.Axis(titleColor='#E0E0E0', labelColor='#E0E0E0')),
            tooltip=[
                alt.Tooltip('Composite_Resilience_Index', bin=True, title='Index Range'),
                alt.Tooltip('count()', title='Ward Count')
            ]
        ).properties(
            title=alt.Title('Distribution of Composite Resilience Index', anchor='start', fontSize=18, color='#E0E0E0')
        ).configure_axis(
            gridColor='#34495E', domainColor='#34495E', tickColor='#34495E',
        ).configure_view(
            strokeWidth=0, fill='#1A1A1A'
        ).interactive()
        
        st.altair_chart(distribution_chart, use_container_width=True)
    else:
        st.warning("Resilience Index data is not available. Please check the data processing steps.", icon="⚠️")


# --- Footer ---
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; font-size: 0.9em; color: #BDC3C7; padding-top: 15px;'>
        <p>Developed for the Urban Resilience Hackathon - {CURRENT_MONTH_YEAR}</p>
        <p>Data sources: BBMP, Karnataka State Natural Disaster Monitoring Centre (KSNDMC), Open Data Initiatives.</p>
        <p>Powered by Streamlit, GeoPandas, Folium, and Altair.</p>
        <p>&copy; 2025 Team HAURCC. All rights reserved.</p>
        <p style='font-style: italic; font-size: 0.8em; color: #6C7A89;'>
            Disclaimer: All advanced features are demonstrative, derived from static, publicly available data,
            and represent a conceptual framework for a real-world command center.
        </p>
    </div>
    """, unsafe_allow_html=True)
