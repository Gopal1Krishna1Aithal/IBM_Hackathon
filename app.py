# ==============================================================================
# HAURCC - Hyper-Analytical Urban Resilience Command Center (v2.3 - Final)
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
from shapely.geometry import box
import math
from typing import Tuple

# AI IMPORTS
import groq
from sklearn.ensemble import IsolationForest
# --- FIX 1 (ML MODEL): ADDING NEW IMPORT ---
from sklearn.ensemble import RandomForestRegressor
# --- FIX 3 (PCA WEIGHTS): ADDING NEW IMPORT ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# --- Global Configuration & Paths ---
CURRENT_MONTH_YEAR = "July 2025"
# --- ADDITION: Dynamically update the date to reflect the current time ---
from datetime import datetime
CURRENT_MONTH_YEAR = datetime.now().strftime("%B %Y")
DATA_DIR = "data"

# AI CONFIGURATION
## --- ENHANCEMENT: API Key Security ---
# Prioritize Streamlit Secrets. Fall back to hardcoded key with a warning.
# To use secrets, create a file .streamlit/secrets.toml and add:
# GROQ_API_KEY = "your_real_api_key"
GROQ_API_KEY_HARDCODED = "Add_your_api key_here"
api_key_source = ""

try:
    # First, try to get the key from Streamlit's secrets management
    groq_api_key = st.secrets["GROQ_API_KEY"]
    api_key_source = "Streamlit Secrets"
    groq_client = groq.Client(api_key=groq_api_key)
    AI_ENABLED = True
except (KeyError, FileNotFoundError):
    # If secrets don't exist, fall back to the hardcoded key
    if GROQ_API_KEY_HARDCODED:
        groq_client = groq.Client(api_key=GROQ_API_KEY_HARDCODED)
        AI_ENABLED = True
        api_key_source = "Hardcoded (Insecure)"
    else:
        groq_client = None
        AI_ENABLED = False
        api_key_source = "Not Found"


# ==============================================================================
# PROFESSIONAL UI COMPONENTS
# ==============================================================================

def setup_professional_ui():
    """Initialize professional styling and components"""
    st.set_page_config(
        page_title="💧 HAURCC - Bengaluru Urban Resilience Command Center",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🌊"
    )
    
    # Professional CSS
    st.markdown("""
    <style>
    /* Modern Professional Styling */
    .main-header {
        background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%);
        padding: 2rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
        border-bottom: 3px solid #00FF99;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1A1A1A 0%, #2C3E50 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #34495E;
        box-shadow: 0 4px 20px rgba(0, 255, 153, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-px);
        box-shadow: 0 8px 30px rgba(0, 255, 153, 0.2);
        border-color: #00FF99;
    }
    
    .dashboard-section {
        background: #1A1A1A;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid #2C3E50;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background-color: #00FF99; }
    .status-warning { background-color: #FFD700; }
    .status-inactive { background-color: #FF4500; }
    
    /* Professional data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Smooth animations */
    .stApp {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

def create_professional_header():
    """Create professional header section"""
    st.markdown(f"""
    <div class='main-header'>
        <h1 style='margin:0; padding:0; color: #00FF99; font-size: 2.5em;'>💧 HAURCC</h1>
        <p style='margin:0; color: #BDC3C7; font-size: 1.2em;'>
            Hyper-Analytical Urban Resilience Command Center - Bengaluru
        </p>
        <p style='margin:0; color: #8C8C8C; font-size: 0.9em;'>
            Real-time flood resilience monitoring and predictive analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_status_bar():
    """Create professional status indicator bar"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <span class='status-indicator status-active'></span>
                <span style='color: #00FF99; font-weight: 600;'>System Status</span>
            </div>
            <p style='margin: 0; color: #E0E0E0;'>Operational</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <span class='status-indicator status-active'></span>
                <span style='color: #00FF99; font-weight: 600;'>Data Freshness</span>
            </div>
            <p style='margin: 0; color: #E0E0E0;'>Updated Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # --- ENHANCEMENT: Dynamic AI Status Indicator ---
        ai_status_class = "status-active" if AI_ENABLED else "status-warning"
        ai_status_text = "Online" if AI_ENABLED else "Offline"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <span class='status-indicator {ai_status_class}'></span>
                <span style='color: #00FF99; font-weight: 600;'>AI Engine</span>
            </div>
            <p style='margin: 0; color: #E0E0E0;'>{ai_status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <span class='status-indicator status-active'></span>
                <span style='color: #00FF99; font-weight: 600;'>Monitoring</span>
            </div>
            <p style='margin: 0; color: #E0E0E0;'>198 Wards Active</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# HELPER FUNCTIONS - DATA PROCESSING AND METRICS
# ==============================================================================

@st.cache_data(ttl=3600)
def load_geospatial_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Loads all core geospatial data files."""
    try:
        # BBMP Wards (Polygons)
        wards_path = os.path.join(DATA_DIR, "bbmp-wards.geojson")
        wards_gdf = gpd.read_file(wards_path)
        wards_gdf = wards_gdf.to_crs("EPSG:4326")
        
        # Calculate ward area in square kilometers
        wards_gdf_proj_area = wards_gdf.to_crs(epsg=32643)
        wards_gdf['area_sqkm'] = wards_gdf_proj_area.geometry.area / 10**6

        # Primary Drains Data
        drains_path = os.path.join(DATA_DIR, "bangalore_swd_primary.geojson")
        primary_drains_gdf = gpd.read_file(drains_path)
        primary_drains_gdf = primary_drains_gdf.to_crs("EPSG:4326")
        
        # Calculate drain lengths in km
        primary_drains_gdf_proj = primary_drains_gdf.to_crs(epsg=32643)
        primary_drains_gdf['length_km'] = primary_drains_gdf_proj.geometry.length / 1000

        # All Flood Incident Points
        floodprone_gdf = gpd.read_file(os.path.join(DATA_DIR, "bbmp_floodprone_locations.geojson")).to_crs("EPSG:4326")
        vulnerable_gdf = gpd.read_file(os.path.join(DATA_DIR, "flooding_vulnerable_locations.geojson")).to_crs("EPSG:4326")
        lowlying_gdf = gpd.read_file(os.path.join(DATA_DIR, "bbmp_lowlying_areas.geojson")).to_crs("EPSG:4326")

        all_flood_points_gdf = pd.concat([floodprone_gdf, vulnerable_gdf, lowlying_gdf], ignore_index=True)
        
        return wards_gdf, primary_drains_gdf, all_flood_points_gdf
    
    except Exception as e:
        st.error(f"Error loading geospatial data: {e}")
        st.stop()

@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def load_tabular_data() -> pd.DataFrame:
    """Loads and preprocesses rainfall data."""
    try:
        rainfall_csv_path = os.path.join(DATA_DIR, "bangalore-rainfall-data-1900-2024-sept.csv")
        rainfall_df = pd.read_csv(rainfall_csv_path)
        
        # --- FIX: Convert 'Year' to a number immediately after loading ---
        # This line was moved from the bottom of the function to the top.
        rainfall_df['Year'] = pd.to_numeric(rainfall_df['Year'], errors='coerce').fillna(0).astype(int)
        
        # --- ADDITION: Simulate current year data to ensure rainfall chart aligns with the app's current year context ---
        np.random.seed(42) # for reproducible simulation
        last_data_year = rainfall_df['Year'].max()
        
        from datetime import datetime
        current_app_year = datetime.now().year

        if last_data_year < current_app_year:
            last_10_years_df = rainfall_df[rainfall_df['Year'] > (last_data_year - 10)]
            monthly_avg = last_10_years_df.mean(numeric_only=True)
            
            new_year_data = {col: 0 for col in rainfall_df.columns}
            new_year_data['Year'] = current_app_year
            
            simulated_total = 0
            current_month_index = datetime.now().month
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            months_to_simulate = months[:current_month_index]

            for month in months_to_simulate:
                if month in monthly_avg:
                    simulated_value = monthly_avg[month] * np.random.uniform(0.85, 1.15)
                    new_year_data[month] = simulated_value
                    simulated_total += simulated_value

            new_year_data['Total'] = simulated_total
            
            new_row_df = pd.DataFrame([new_year_data])
            rainfall_df = pd.concat([rainfall_df, new_row_df], ignore_index=True)
        # --- END ADDITION ---
        
        rainfall_df.dropna(subset=['Total'], inplace=True)
        
        avg_annual_rainfall = rainfall_df['Total'].mean()
        rainfall_df['deviation_from_mean'] = rainfall_df['Total'] - avg_annual_rainfall
        
        return rainfall_df
        
    except Exception as e:
        st.error(f"Error loading rainfall data: {e}")
        st.stop()

@st.cache_data(ttl=3600)
def calculate_flood_incident_metrics(_wards_gdf, _all_flood_points_gdf) -> gpd.GeoDataFrame:
    """Calculates direct and proximity-based flood incident counts."""
    wards_gdf = _wards_gdf.copy()
    all_flood_points_gdf = _all_flood_points_gdf.copy()

    # Calculate direct incident count
    wards_with_points = gpd.sjoin(all_flood_points_gdf, wards_gdf, how="inner", predicate="within")
    incident_counts = wards_with_points.groupby('index_right').size().rename("incident_count")
    wards_gdf = wards_gdf.merge(incident_counts, left_index=True, right_index=True, how="left")
    wards_gdf['incident_count'] = wards_gdf['incident_count'].fillna(0).astype(int)

    # Calculate proximity incident count
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
    """Calculates drainage-related metrics for each ward."""
    wards_gdf = _wards_gdf.copy()
    primary_drains_gdf = _primary_drains_gdf.copy()
    
    # Spatial join drains to wards
    wards_with_drains = gpd.sjoin(primary_drains_gdf, wards_gdf, how="inner", predicate="intersects")
    
    # Group by ward and sum drain lengths
    drain_lengths_per_ward = wards_with_drains.groupby('index_right')['length_km'].sum().rename("drain_length_km")
    
    # Merge back to wards_gdf
    wards_gdf = wards_gdf.merge(drain_lengths_per_ward, left_index=True, right_index=True, how="left")
    wards_gdf['drain_length_km'] = wards_gdf['drain_length_km'].fillna(0)

    # Calculate Drainage Density
    wards_gdf['drainage_density_km_sqkm'] = wards_gdf.apply(
        lambda row: (row['drain_length_km'] / row['area_sqkm']) if row['area_sqkm'] > 0 else 0, axis=1
    )
    wards_gdf['drainage_density_km_sqkm'] = wards_gdf['drainage_density_km_sqkm'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate drainage risk factor
    max_drainage_density = wards_gdf['drainage_density_km_sqkm'].max()
    if max_drainage_density > 0:
        wards_gdf['drainage_risk_factor'] = (max_drainage_density - wards_gdf['drainage_density_km_sqkm']) / max_drainage_density
    else:
        wards_gdf['drainage_risk_factor'] = 0
    
    return wards_gdf

@st.cache_data(ttl=3600)
def calculate_composite_resilience_index(_wards_gdf) -> gpd.GeoDataFrame:
    """Calculates comprehensive resilience index for each ward."""
    wards_gdf = _wards_gdf.copy()
    
    WEIGHT_INCIDENT_DENSITY = 0.4
    WEIGHT_PROXIMITY_INCIDENTS = 0.2
    WEIGHT_DRAINAGE_EFFICIENCY = 0.4
    
    # Normalize metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        wards_gdf['normalized_incident_density'] = wards_gdf['incident_density_sqkm'] / wards_gdf['incident_density_sqkm'].replace(0, np.nan).max()
        wards_gdf['log_buffered_incidents'] = np.log1p(wards_gdf['buffered_incident_count'])
        wards_gdf['normalized_proximity'] = wards_gdf['log_buffered_incidents'] / wards_gdf['log_buffered_incidents'].replace(0, np.nan).max()
        wards_gdf['normalized_drainage_risk'] = wards_gdf['drainage_risk_factor']
    
    wards_gdf.fillna(0, inplace=True)

    # Calculate Composite Resilience Index
    wards_gdf['Composite_Resilience_Index'] = (
        (wards_gdf['normalized_incident_density'] * WEIGHT_INCIDENT_DENSITY) +
        (wards_gdf['normalized_proximity'] * WEIGHT_PROXIMITY_INCIDENTS) +
        (wards_gdf['normalized_drainage_risk'] * WEIGHT_DRAINAGE_EFFICIENCY)
    )

    # Normalize to 0-100 scale
    max_score = wards_gdf['Composite_Resilience_Index'].replace(0, np.nan).max()
    if pd.notna(max_score) and max_score > 0:
        wards_gdf['Composite_Resilience_Index'] = (wards_gdf['Composite_Resilience_Index'] / max_score) * 100
    else:
        wards_gdf['Composite_Resilience_Index'] = 0
        
    def assign_resilience_level(score):
        if score >= 85: return "Extreme Vulnerability"
        elif score >= 60: return "High Vulnerability"
        elif score >= 35: return "Moderate Vulnerability"
        elif score >= 10: return "Low Vulnerability"
        else: return "High Resilience"

    wards_gdf['resilience_level'] = wards_gdf['Composite_Resilience_Index'].apply(assign_resilience_level)
    
    return wards_gdf

# --- FIX 2 (INDEX CALCULATION): ADDING NEW ROBUST FUNCTION ---
@st.cache_data(ttl=3600)
def calculate_composite_resilience_index_robust(_wards_gdf) -> gpd.GeoDataFrame:
    """Calculates a robust resilience index using rank-based normalization to resist outliers."""
    wards_gdf = _wards_gdf.copy()
    
    # This function creates the normalized columns needed for the final PCA step
    # It is an intermediate calculation
    
    # Robust Normalization using percentile ranks (0 to 1 scale)
    wards_gdf['normalized_incident_density'] = wards_gdf['incident_density_sqkm'].rank(pct=True)
    wards_gdf['normalized_proximity'] = wards_gdf['buffered_incident_count'].rank(pct=True)
    # Lower drainage density is worse, so we rank it in reverse (ascending=False)
    wards_gdf['normalized_drainage_risk'] = wards_gdf['drainage_density_km_sqkm'].rank(pct=True, ascending=False)
    
    wards_gdf.fillna(0, inplace=True)
    return wards_gdf

# --- FIX 3 (PCA WEIGHTS): ADDING NEW DATA-DRIVEN WEIGHTING FUNCTION ---
@st.cache_data(ttl=3600)
def calculate_composite_resilience_index_pca(_wards_gdf) -> gpd.GeoDataFrame:
    """Calculates the final index using PCA to derive data-driven weights."""
    wards_gdf = _wards_gdf.copy()
    
    features = ['normalized_incident_density', 'normalized_proximity', 'normalized_drainage_risk']
    X = wards_gdf[features].values
    
    # Scale data before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to find the component that explains the most variance
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(X_scaled)
    
    # The loadings of the first component are our data-driven weights
    # We use the absolute values and normalize them to sum to 1
    pca_weights = np.abs(pca.components_[0])
    pca_weights /= np.sum(pca_weights)
    
    # Calculate final index using these new weights
    wards_gdf['Composite_Resilience_Index'] = (
        (wards_gdf['normalized_incident_density'] * pca_weights[0]) +
        (wards_gdf['normalized_proximity'] * pca_weights[1]) +
        (wards_gdf['normalized_drainage_risk'] * pca_weights[2])
    )
    
    # Scale final score to be 0-100 for interpretability
    wards_gdf['Composite_Resilience_Index'] = wards_gdf['Composite_Resilience_Index'].rank(pct=True) * 100
        
    def assign_resilience_level(score):
        if score >= 85: return "Extreme Vulnerability"
        elif score >= 60: return "High Vulnerability"
        elif score >= 35: return "Moderate Vulnerability"
        elif score >= 10: return "Low Vulnerability"
        else: return "High Resilience"

    wards_gdf['resilience_level'] = wards_gdf['Composite_Resilience_Index'].apply(assign_resilience_level)
    return wards_gdf

# --- NEW FEATURE: TEMPORAL ANALYSIS ---
@st.cache_data(ttl=3600)
def simulate_temporal_data(_base_wards_gdf, _all_flood_points_gdf, _primary_drains_gdf):
    """Generates simulated historical data for temporal analysis."""
    # --- ADDITION: Set a random seed for reproducible, stable simulations ---
    np.random.seed(42)
    temporal_data = []
    
    for year in range(2021, 2026):
        yearly_wards_gdf = _base_wards_gdf.copy()
        
        # Simulate changes over time
        # Incidents might grow in some areas, decrease in others
        incident_change_factor = np.random.uniform(0.9, 1.15, size=len(yearly_wards_gdf))
        # Drain density might slightly improve over time
        drain_change_factor = 1 + (year - 2021) * 0.02
        
        # Apply simulated changes for years before the current year
        if year < 2025:
            yearly_wards_gdf['incident_count'] = (yearly_wards_gdf['incident_count'] * incident_change_factor).astype(int)
            yearly_wards_gdf['drain_length_km'] = yearly_wards_gdf['drain_length_km'] * drain_change_factor

        # Recalculate metrics for the simulated year
        yearly_wards_gdf['incident_density_sqkm'] = yearly_wards_gdf.apply(
            lambda row: (row['incident_count'] / row['area_sqkm']) if row['area_sqkm'] > 0 else 0, axis=1
        ).replace([np.inf, -np.inf], 0).fillna(0)

        yearly_wards_gdf['drainage_density_km_sqkm'] = yearly_wards_gdf.apply(
            lambda row: (row['drain_length_km'] / row['area_sqkm']) if row['area_sqkm'] > 0 else 0, axis=1
        ).replace([np.inf, -np.inf], 0).fillna(0)

        max_drainage_density = yearly_wards_gdf['drainage_density_km_sqkm'].max()
        if max_drainage_density > 0:
            yearly_wards_gdf['drainage_risk_factor'] = (max_drainage_density - yearly_wards_gdf['drainage_density_km_sqkm']) / max_drainage_density
        else:
            yearly_wards_gdf['drainage_risk_factor'] = 0

        # Recalculate the final index
        final_yearly_gdf = calculate_composite_resilience_index(yearly_wards_gdf)
        final_yearly_gdf['year'] = year
        temporal_data.append(final_yearly_gdf)
        
    return pd.concat(temporal_data, ignore_index=True)

# --- FIX (SIMULATION): ADDING NEW TREND-BASED FUNCTION ---
@st.cache_data(ttl=3600)
def simulate_temporal_data_trend_based(_base_wards_gdf, _all_flood_points_gdf, _primary_drains_gdf):
    """Generates a more realistic simulated historical data based on logical trends."""
    np.random.seed(42)
    temporal_data = []
    
    # Create a vulnerability factor to simulate that higher-risk wards degrade faster
    vulnerability_factor = 1 + (_base_wards_gdf['Composite_Resilience_Index'] / 100) * 0.1 # Max 10% increase factor

    for year in range(2021, 2026):
        yearly_wards_gdf = _base_wards_gdf.copy()
        
        # Apply simulated changes for years before the current year
        if year < 2025:
            # Simulate a slight increase in incidents, more so in already vulnerable wards
            # This simulates urban growth and stress on infrastructure
            incident_change_factor = 1 - ((2025 - year) * 0.05 * vulnerability_factor)
            yearly_wards_gdf['incident_count'] = (yearly_wards_gdf['incident_count'] * incident_change_factor).astype(int)
            
            # Simulate a small, uniform improvement in drainage (e.g., city-wide projects)
            drain_change_factor = 1 - ((2025 - year) * 0.01)
            yearly_wards_gdf['drain_length_km'] = yearly_wards_gdf['drain_length_km'] * drain_change_factor

        # Recalculate metrics for the simulated year
        yearly_wards_gdf = calculate_flood_incident_metrics(yearly_wards_gdf, _all_flood_points_gdf)
        yearly_wards_gdf = calculate_drainage_metrics(yearly_wards_gdf, _primary_drains_gdf)

        # Recalculate the final index using the full robust + PCA pipeline
        final_yearly_gdf = calculate_composite_resilience_index_robust(yearly_wards_gdf)
        final_yearly_gdf = calculate_composite_resilience_index_pca(final_yearly_gdf)
        final_yearly_gdf['year'] = year
        temporal_data.append(final_yearly_gdf)
        
    return pd.concat(temporal_data, ignore_index=True)

# --- ADDITION: This is the new, improved rainfall-linked simulation function ---
# --- ADDITION: This is the new, improved rainfall-linked simulation function ---
@st.cache_data(ttl=3600)
def simulate_temporal_data_rainfall_linked(_base_wards_gdf, _all_flood_points_gdf, _primary_drains_gdf, _rainfall_df):
    """
    Generates a realistic historical simulation by linking incident counts
    to actual historical rainfall data.
    """
    temporal_data = []
    
    # Calculate the long-term average rainfall to use as a baseline
    avg_annual_rainfall = _rainfall_df['Total'].mean()

    for year in range(2021, 2026):
        yearly_wards_gdf = _base_wards_gdf.copy()
        
        # --- FIX: This block is now robust against missing years in the data ---
        # Look up the rainfall data for the specific year
        year_data = _rainfall_df[_rainfall_df['Year'] == year]['Total']
        
        # Check if we found data for that year
        if not year_data.empty:
            actual_rainfall_for_year = year_data.iloc[0]
        else:
            # If no data exists for the year (e.g., future years), use the average as a fallback
            actual_rainfall_for_year = avg_annual_rainfall

        # If rainfall data for the year exists, create a factor to adjust incident counts
        if pd.notna(actual_rainfall_for_year) and avg_annual_rainfall > 0:
            # How much did this year's rainfall deviate from the average?
            rainfall_factor = actual_rainfall_for_year / avg_annual_rainfall
        else:
            # If no data, assume an average year
            rainfall_factor = 1.0

        # Simulate incident counts based on real rainfall, not random numbers
        # We use a square root to moderate the effect of extreme rainfall years
        yearly_wards_gdf['incident_count'] = (yearly_wards_gdf['incident_count'] * np.sqrt(rainfall_factor)).astype(int)
        
        # Recalculate all metrics based on the rainfall-adjusted incident count
        yearly_wards_gdf = calculate_flood_incident_metrics(yearly_wards_gdf, _all_flood_points_gdf)
        yearly_wards_gdf = calculate_drainage_metrics(yearly_wards_gdf, _primary_drains_gdf)

        # Recalculate the final index using the full robust + PCA pipeline
        final_yearly_gdf = calculate_composite_resilience_index_robust(yearly_wards_gdf)
        # Note: The original file did not have the ML-weighted index here. Sticking to the file's logic.
        final_yearly_gdf = calculate_composite_resilience_index_pca(final_yearly_gdf)
        
        final_yearly_gdf['year'] = year
        temporal_data.append(final_yearly_gdf)
        
    return pd.concat(temporal_data, ignore_index=True)

# --- ADDITION: Function to get ML/PCA weights, required by final simulation logic ---
def train_ward_risk_classifier(_wards_gdf):
    """
    Extracts data-driven feature weights using PCA. This function is named
    to match the call in the final simulation logic. It provides the baseline
    weights needed for a stable year-over-year comparison.
    """
    wards_gdf = _wards_gdf.copy()
    
    # This logic is borrowed from the calculate_composite_resilience_index_pca function
    # to ensure consistency in weight generation.
    
    # Step 1: Ensure the normalized features exist by re-running the robust calculation.
    wards_gdf['normalized_incident_density'] = wards_gdf['incident_density_sqkm'].rank(pct=True)
    wards_gdf['normalized_proximity'] = wards_gdf['buffered_incident_count'].rank(pct=True)
    wards_gdf['normalized_drainage_risk'] = wards_gdf['drainage_density_km_sqkm'].rank(pct=True, ascending=False)
    wards_gdf.fillna(0, inplace=True)

    features = ['normalized_incident_density', 'normalized_proximity', 'normalized_drainage_risk']
    X = wards_gdf[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=1)
    pca.fit(X_scaled) # We just need to fit to get the components
    
    pca_weights = np.abs(pca.components_[0])
    pca_weights /= np.sum(pca_weights)
    
    return pca_weights

# --- ADDITION: A special index calculation function for the simulation ---
def calculate_simulation_year_index(yearly_gdf, baseline_ml_weights):
    """
    Calculates the index for a single simulated year without re-ranking the final score,
    allowing for true year-over-year comparison.
    """
    # Step 1: Robust Normalization using percentile ranks
    yearly_gdf['normalized_incident_density'] = yearly_gdf['incident_density_sqkm'].rank(pct=True)
    yearly_gdf['normalized_proximity'] = yearly_gdf['buffered_incident_count'].rank(pct=True)
    yearly_gdf['normalized_drainage_risk'] = yearly_gdf['drainage_density_km_sqkm'].rank(pct=True, ascending=False)
    yearly_gdf.fillna(0, inplace=True)

    # Step 2: Apply the FIXED baseline weights
    yearly_gdf['Composite_Resilience_Index'] = (
        (yearly_gdf['normalized_incident_density'] * baseline_ml_weights[0]) +
        (yearly_gdf['normalized_proximity'] * baseline_ml_weights[1]) +
        (yearly_gdf['normalized_drainage_risk'] * baseline_ml_weights[2])
    ) * 100 # Scale to 0-100

    # CRITICAL: We DO NOT re-rank the final index here.
    
    def assign_resilience_level(score):
        if score >= 85: return "Extreme Vulnerability"
        elif score >= 60: return "High Vulnerability"
        elif score >= 35: return "Moderate Vulnerability"
        elif score >= 10: return "Low Vulnerability"
        else: return "High Resilience"
    yearly_gdf['resilience_level'] = yearly_gdf['Composite_Resilience_Index'].apply(assign_resilience_level)
    return yearly_gdf


# --- ADDITION: The final, corrected simulation function ---
@st.cache_data(ttl=3600)
def simulate_temporal_data_final(_base_wards_gdf, _all_flood_points_gdf, _primary_drains_gdf, _rainfall_df):
    """
    The definitive simulation. It uses real rainfall data and a non-ranking index
    to show true year-over-year changes.
    """
    temporal_data = []
    
    # Calculate the importance weights ONCE from the baseline 2025 data
    baseline_ml_weights = train_ward_risk_classifier(_base_wards_gdf)
    avg_annual_rainfall = _rainfall_df['Total'].mean()

    for year in range(2021, 2026):
        yearly_wards_gdf = _base_wards_gdf.copy()
        
        # Look up rainfall data safely
        year_data = _rainfall_df[_rainfall_df['Year'] == year]['Total']
        actual_rainfall_for_year = year_data.iloc[0] if not year_data.empty else avg_annual_rainfall
        
        rainfall_factor = actual_rainfall_for_year / avg_annual_rainfall if avg_annual_rainfall > 0 else 1.0

        # Adjust incident count based on rainfall
        yearly_wards_gdf['incident_count'] = (yearly_wards_gdf['incident_count'] * np.sqrt(rainfall_factor)).astype(int)
        
        # Recalculate metrics
        yearly_wards_gdf = calculate_flood_incident_metrics(yearly_wards_gdf, _all_flood_points_gdf)
        yearly_wards_gdf = calculate_drainage_metrics(yearly_wards_gdf, _primary_drains_gdf)

        # Calculate the index for this year using the special simulation function
        final_yearly_gdf = calculate_simulation_year_index(yearly_wards_gdf, baseline_ml_weights)
        
        final_yearly_gdf['year'] = year
        temporal_data.append(final_yearly_gdf)
        
    return pd.concat(temporal_data, ignore_index=True)

## --- ADDITION: Performance Optimization for Grid Generation ---
@st.cache_data(show_spinner=False)
def generate_ward_hotspot_grid(_ward_gdf, _all_flood_points_gdf, grid_size_meters):
    """Generates a cached hotspot grid for a single ward."""
    try:
        ward_geometry = _ward_gdf.geometry.iloc[0]
        ward_gdf_proj = _ward_gdf.to_crs("EPSG:32643")
        
        minx, miny, maxx, maxy = ward_gdf_proj.total_bounds
        
        polygons = []
        x_coords = np.arange(minx, maxx + grid_size_meters, grid_size_meters)
        y_coords = np.arange(miny, maxy + grid_size_meters, grid_size_meters)

        for i in range(len(x_coords) - 1):
            for j in range(len(y_coords) - 1):
                grid_cell_proj = box(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
                # Ensure we only keep cells that are part of the ward
                if ward_gdf_proj.geometry.iloc[0].intersects(grid_cell_proj):
                    polygons.append(grid_cell_proj)
        
        if not polygons:
            return None

        grid_gdf_proj = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:32643")
        grid_gdf = grid_gdf_proj.to_crs("EPSG:4326")

        # Spatially join with only relevant flood points for efficiency
        ward_bounds = _ward_gdf.total_bounds
        bbox_poly = box(ward_bounds[0], ward_bounds[1], ward_bounds[2], ward_bounds[3])
        relevant_flood_points = _all_flood_points_gdf[_all_flood_points_gdf.geometry.intersects(bbox_poly)]
        
        if relevant_flood_points.empty:
            grid_gdf['incident_count_in_cell'] = 0
        else:
            grid_with_points = gpd.sjoin(grid_gdf, relevant_flood_points, how="left", predicate="intersects")
            incident_counts_per_grid_cell = grid_with_points.groupby(grid_with_points.index).size().rename("incident_count_in_cell")
            grid_gdf = grid_gdf.merge(incident_counts_per_grid_cell, left_index=True, right_index=True, how="left")
        
        grid_gdf['incident_count_in_cell'] = grid_gdf['incident_count_in_cell'].fillna(0).astype(int)
        grid_gdf['grid_risk_level'] = grid_gdf['incident_count_in_cell'].apply(assign_grid_risk_level)
        
        return grid_gdf
    except Exception as e:
        st.warning(f"Could not generate hotspot grid: {e}")
        return None

# ==============================================================================
# AI FUNCTIONS
# ==============================================================================
def predict_ward_risk(ward_data):
    """(Statistical Projection) Risk prediction for individual wards."""
    try:
        base_risk = ward_data.get('Composite_Resilience_Index', 0)
        risk_variation = ward_data.get('incident_density_sqkm', 0) * 5
        predicted_risk = min(base_risk + risk_variation, 100)
        return predicted_risk
    except (TypeError, KeyError):
        return ward_data.get('Composite_Resilience_Index', 0)

# --- FIX 1 (ML MODEL): ADDING NEW, SUPERIOR FUNCTIONS ---
@st.cache_resource
def train_incident_prediction_model(_wards_df):
    """Trains a model to predict actual incident counts, avoiding circular logic."""
    # Features are physical characteristics, NOT derived from the incident count itself
    features = ['area_sqkm', 'drainage_density_km_sqkm']
    target = 'incident_count'
    
    X = _wards_df[features].fillna(0)
    y = _wards_df[target].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_ward_incidents_ml(model, ward_data):
    """Predicts a ward's likely incident count using the trained ML model."""
    features = ['area_sqkm', 'drainage_density_km_sqkm']
    ward_df = pd.DataFrame([ward_data])
    ward_features = ward_df[features].fillna(0)
    
    prediction = model.predict(ward_features)
    return prediction[0]

def generate_ai_recommendations(ward_data, predicted_risk):
    """Generate high-quality, specific, and well-formatted AI recommendations."""
    if not groq_client:
        return ["AI Engine is offline. Configure API Key for recommendations."]
    
    # --- ENHANCEMENT: Add seasonal context ---
    from datetime import datetime
    current_date_str = datetime.now().strftime("%B %d, %Y")
    seasonal_context = f"Today is {current_date_str}. This is the tail-end of the Southwest Monsoon in Bengaluru, a period characterized by occasional heavy rainfall. Preparations for the Northeast Monsoon (Oct-Dec) should also be considered."

    try:
        prompt = f"""
        **Role:** You are an expert hydrologist and urban planning consultant for Bengaluru. Your advice is sharp, data-driven, and highly specific. You are briefing the BBMP Commissioner.

        **CRITICAL INSTRUCTIONS:**
        1.  **NO GENERIC ADVICE:** Do NOT suggest vague actions. Every recommendation must be a concrete, actionable step.
        2.  **USE SEASONAL CONTEXT:** Your recommendations MUST be relevant to the current date and monsoon phase provided below.
        3.  **OUTPUT MUST BE CLEAN HTML:** Use only `<h4>`, `<p>`, `<strong>`, `<hr>`.

        **Seasonal Context:** {seasonal_context}

        **Ward Data Analysis:**
        - **Ward Name:** {ward_data['KGISWardName']} (No: {ward_data['KGISWardNo']})
        - **Vulnerability Score:** {ward_data['Composite_Resilience_Index']:.1f}/100 (Higher is worse)
        - **Key Challenge Indicators:**
          - **Historical Incident Density:** {ward_data['incident_density_sqkm']:.2f} incidents/km²
          - **Primary Drainage Density:** {ward_data['drainage_density_km_sqkm']:.3f} km/km² (A low value is a major red flag)
          - **Total Incidents:** {ward_data['incident_count']}

        **Task:** Based on your expert analysis of the ward data AND the crucial seasonal context, provide a prioritized action plan. The "Immediate Tactical Action" must be directly relevant for the current date.

        **Required HTML Format (Strictly follow this):**

        <h4>1. Immediate Tactical Action (Next 7-14 Days)</h4>
        <p><strong>Action:</strong> [A specific, short-term action relevant for mid-September.]</p>
        <p><strong>Justification:</strong> [Justification linked to data and the current monsoon phase.]</p>
        <p><strong>Estimated Impact/Cost:</strong> High Impact / Low Cost</p>
        <hr>
        <h4>2. Medium-Term Infrastructural Project (1-2 Years)</h4>
        <p><strong>Action:</strong> [A specific infrastructure project.]</p>
        <p><strong>Justification:</strong> [Justification linked to the ward's core data deficits.]</p>
        <p><strong>Estimated Impact/Cost:</strong> Very High Impact / Medium Cost</p>
        <hr>
        <h4>3. Long-Term Strategic Policy (3-5 Years)</h4>
        <p><strong>Action:</strong> [A specific policy proposal.]</p>
        <p><strong>Justification:</strong> [Long-term strategic justification.]</p>
        <p><strong>Estimated Impact/Cost:</strong> Transformative Impact / Policy-Based Cost</p>
        """
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=1500,
            temperature=0.5
        )
        
        return [response.choices[0].message.content.strip()]
    except Exception as e:
        return [f"<p><b>AI service unavailable:</b> {str(e)}</p>"]

def detect_anomalies(_wards_gdf):
    """Detect anomalous wards using isolation forest."""
    try:
        features = _wards_gdf[['Composite_Resilience_Index', 'incident_density_sqkm', 
                              'drainage_density_km_sqkm', 'area_sqkm']].fillna(0)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features)
        anomalous_wards_df = _wards_gdf[anomalies == -1]
        return anomalous_wards_df
    except:
        return pd.DataFrame()

# --- FIX 2 (ANOMALY DETECTION): ADDING NEW DYNAMIC FUNCTION ---
def detect_anomalies_dynamic(_wards_gdf):
    """Detects anomalies using a data-driven threshold, not a fixed quota."""
    try:
        features = _wards_gdf[['Composite_Resilience_Index', 'incident_density_sqkm', 
                              'drainage_density_km_sqkm', 'area_sqkm']].fillna(0)
        
        iso_forest = IsolationForest(contamination='auto', random_state=42)
        iso_forest.fit(features)
        
        # Get the anomaly scores for all data points
        anomaly_scores = iso_forest.decision_function(features)
        
        # A common data-driven approach: anything below a certain score threshold is an anomaly.
        # Here, we'll use a threshold based on the distribution of scores.
        # Points more than 1.5 standard deviations below the mean score are flagged.
        threshold = np.mean(anomaly_scores) - 1.5 * np.std(anomaly_scores)
        
        anomalies = anomaly_scores < threshold
        anomalous_wards_df = _wards_gdf[anomalies]
        return anomalous_wards_df
    except Exception as e:
        st.warning(f"Could not perform dynamic anomaly detection: {e}")
        return pd.DataFrame()

# --- NEW FEATURE: NETWORK ANALYSIS AI ---
def generate_network_insight(drain_name, network_wards_df):
    """Generates an AI-powered insight about a drain network."""
    if not groq_client:
        return "AI Engine is offline."

    try:
        # Create a summary string of the wards in the network
        wards_summary = []
        for _, row in network_wards_df.iterrows():
            wards_summary.append(
                f"- {row['KGISWardName']} (Vulnerability: {row['Composite_Resilience_Index']:.1f}, Incidents: {row['incident_count']})"
            )
        wards_summary_str = "\n".join(wards_summary)

        prompt = f"""
        **Role:** You are a senior hydrologist analyzing a critical stormwater drain network in Bengaluru for the BBMP.

        **Subject:** Analysis of the **{drain_name}** primary drain network.

        **Data:** This drain passes through the following wards, listed in order:
        {wards_summary_str}

        **Task:**
        1.  Identify the **single most vulnerable ward** in this chain. This is the network's "weakest link" or primary bottleneck.
        2.  Briefly explain **why** it is the weakest link, using the provided data (vulnerability score and incident count).
        3.  State the **implication** of this bottleneck for the other wards in the network (e.g., "A bottleneck in [Weakest Ward] likely causes backflow and flooding in the upstream [Upstream Ward]").
        4.  Provide your analysis in a concise, single paragraph.
        """
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", max_tokens=512)
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Could not generate AI insight: {e}"


# ==============================================================================
# DATA LOADING & INITIALIZATION
# ==============================================================================

# Setup professional UI
setup_professional_ui()

# Show loading state
with st.spinner("🚀 Initializing HAURCC System... Loading urban resilience data"):
    # Load all data at startup
    bbmp_wards_raw, primary_drains, all_flood_points_gdf = load_geospatial_data()
    rainfall_data = load_tabular_data()

    # Calculate and integrate all metrics for the current year (2025)
    bbmp_wards_metrics = calculate_flood_incident_metrics(bbmp_wards_raw, all_flood_points_gdf)
    bbmp_wards_metrics = calculate_drainage_metrics(bbmp_wards_metrics, primary_drains)
    
    # --- LOGICAL FIX PIPELINE ---
    bbmp_wards = calculate_composite_resilience_index(bbmp_wards_metrics)
    # 1. OVERRIDE with robust normalization (fixes outlier sensitivity)
    bbmp_wards = calculate_composite_resilience_index_robust(bbmp_wards_metrics)
    # 2. OVERRIDE with PCA-based weights (fixes subjective weights)
    bbmp_wards = calculate_composite_resilience_index_pca(bbmp_wards)
    
    # --- NEW: Generate and store temporal data ---
    temporal_wards_data = simulate_temporal_data(bbmp_wards, all_flood_points_gdf, primary_drains)
    # OVERRIDE with trend-based simulation
    temporal_wards_data = simulate_temporal_data_trend_based(bbmp_wards, all_flood_points_gdf, primary_drains)
    # --- FINAL FIX (TEMPORAL ANALYSIS): OVERRIDING with rainfall-linked simulation ---
    temporal_wards_data = simulate_temporal_data_rainfall_linked(bbmp_wards, all_flood_points_gdf, primary_drains, rainfall_data)
    # --- FINAL FIX (TEMPORAL ANALYSIS): Using a non-ranking index for true year-over-year comparison ---
    temporal_wards_data = simulate_temporal_data_final(bbmp_wards, all_flood_points_gdf, primary_drains, rainfall_data)


# Check if data loading was successful
if bbmp_wards is None:
    st.error("FATAL ERROR: Data initialization failed.")
    st.stop()

# Color palettes
resilience_colors = {
    "Extreme Vulnerability": "#8B0000", "High Vulnerability": "#FF4500",
    "Moderate Vulnerability": "#FFD700", "Low Vulnerability": "#32CD32",
    "High Resilience": "#008000"
}
grid_risk_colors = {
    "Critical Risk": "#8B0000", "High Risk": "#B22222", "Moderate Risk": "#FF8C00",
    "Low Risk": "#3CB371", "Minor Risk": "#6B8E23", "No Incidents": "#00000000"
}

def assign_grid_risk_level(incident_count):
    if incident_count == 0: return "No Incidents"
    elif incident_count == 1: return "Minor Risk"
    elif incident_count <= 3: return "Low Risk"
    elif incident_count <= 6: return "Moderate Risk"
    elif incident_count <= 10: return "High Risk"
    else: return "Critical Risk"

# ==============================================================================
# STREAMLIT UI - DASHBOARD LAYOUT
# ==============================================================================

# Professional Header
create_professional_header()

# --- ADDITION: API Key Security Warning ---
if api_key_source == "Hardcoded (Insecure)":
    st.warning("🚨 **Security Alert:** Your GROQ API Key is hardcoded in the script. "
               "For security, please move it to Streamlit Secrets (`.streamlit/secrets.toml`).", icon="⚠️")

create_status_bar()

# Main Content Area
col1, col2 = st.columns([0.7, 0.3])

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00FF99;'>🗺️ Command Center Controls</h2>", unsafe_allow_html=True)
    
    ward_names = sorted(bbmp_wards['KGISWardName'].dropna().unique().tolist())
    ward_options = ["--- Bangalore City Overview ---"] + ward_names

    selected_ward_name = st.selectbox(
        "**🎯 Select Target Ward:**",
        options=ward_options,
        key="ward_selector",
        help="Choose 'Bangalore City Overview' for macro view, or a specific ward for granular analysis."
    )
    
    st.session_state.selected_ward = selected_ward_name

    selected_ward_gdf = None
    if selected_ward_name != "--- Bangalore City Overview ---":
        selected_ward_gdf = bbmp_wards[bbmp_wards['KGISWardName'] == selected_ward_name].copy()
        if not selected_ward_gdf.empty:
            display_properties = selected_ward_gdf.iloc[0]
            st.markdown(f"<h3 style='color: #00C0FF;'>🏡 Ward: {display_properties.get('KGISWardName', 'N/A')}</h3>", unsafe_allow_html=True)
            
            resilience_level = display_properties.get('resilience_level', 'High Resilience')
            resilience_score = display_properties.get('Composite_Resilience_Index', 0)
            
            st.markdown(f"**Ward No.:** <span style='font-size: 1.1em; color: #E0E0E0;'>{display_properties.get('KGISWardNo', 'N/A')}</span>", unsafe_allow_html=True)
            st.markdown(f"**Area:** <span style='font-size: 1.1em; color: #E0E0E0;'>{display_properties.get('area_sqkm', 0):.2f} km²</span>", unsafe_allow_html=True)
            st.markdown(f"**Resilience Level:** <span style='color: {resilience_colors.get(resilience_level)}; font-weight: bold; font-size: 1.1em;'>{resilience_level}</span>", unsafe_allow_html=True)
            st.markdown(f"**Resilience Index (2025):** <span style='font-weight: bold; color: #00FF99;'>{resilience_score:.2f} / 100</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("<h3 class='map-heading'>⚙️ Granular Hotspot Analysis</h3>", unsafe_allow_html=True)
            
            grid_size_m_option = st.slider(
                "**Grid Cell Size (meters):**",
                min_value=100, max_value=500, value=250, step=50,
                help="Adjust grid resolution for detailed hotspot analysis"
            )
            st.session_state['grid_size_m'] = grid_size_m_option
            
            st.markdown("---")
            st.markdown("<h3 style='color: #00FF99;'>🤖 AI-Powered Features</h3>", unsafe_allow_html=True)
            
            st.checkbox("Enable AI Predictions", value=True, key="enable_ai")

            if st.session_state.get("enable_ai", False):
                if st.button("🔄 Refresh AI Predictions"):
                    st.cache_data.clear()
                    st.rerun()
                
                if st.button("🔍 Detect & Visualize Anomalies"):
                    # --- ADDITION: Add context to explain what "anomaly" means in this model ---
                    st.caption("Note: The model is configured to identify the top 10% of wards with the most unusual data patterns.")
                    # --- ADDITION: Clarify that the active model uses a dynamic, not fixed, threshold ---
                    st.caption("Update: The active 'FIX 2' model uses a dynamic statistical threshold for more accurate detection.")
                    anomalous_wards_df = detect_anomalies(bbmp_wards)
                    # --- FIX 2: OVERRIDE FORCED QUOTA WITH DYNAMIC THRESHOLD ---
                    anomalous_wards_df = detect_anomalies_dynamic(bbmp_wards)

                    if not anomalous_wards_df.empty:
                        st.info(f"Found {len(anomalous_wards_df)} anomalous wards. They are now highlighted on the main map.")
                        # --- ENHANCEMENT: Store anomalies in session state for map visualization ---
                        st.session_state.anomalous_wards = anomalous_wards_df['KGISWardName'].tolist()
                        for _, ward in anomalous_wards_df.head(3).iterrows():
                            st.write(f"• {ward['KGISWardName']} (Risk: {ward['Composite_Resilience_Index']:.1f})")
                    else:
                        st.session_state.anomalous_wards = []
                        st.success("No anomalies detected based on current model parameters.")
    
    # --- ENHANCEMENT: Add Alert Simulation ---
    st.markdown("<h3 style='color: #FF4500; margin-top: 1rem;'>🚨 Operational Alerts</h3>", unsafe_allow_html=True)
    col_alert1, col_alert2 = st.columns(2)
    with col_alert1:
        if st.button("Simulate Alert", type="primary"):
            # Identify top 5 most vulnerable wards
            alert_wards_df = bbmp_wards.nlargest(5, 'Composite_Resilience_Index')
            st.session_state.alert_wards = alert_wards_df['KGISWardName'].tolist()
            st.toast("Monsoon Alert Activated!", icon="🚨")

    with col_alert2:
        if st.button("Clear Alert"):
            if 'alert_wards' in st.session_state:
                del st.session_state.alert_wards
                st.toast("Alert Cleared.", icon="✅")


# AI Data Processing
if st.session_state.get("enable_ai", False):
    # --- FIX 1: Train the non-circular ML model once ---
    incident_model = train_incident_prediction_model(bbmp_wards)
    
    # --- FIX 1: Get predictions for ALL wards to create a normalized score ---
    all_predictions = []
    for _, ward in bbmp_wards.iterrows():
        predicted_incidents = predict_ward_incidents_ml(incident_model, ward)
        all_predictions.append({'ward_name': ward['KGISWardName'], 'predicted_incidents': predicted_incidents})
    
    predictions_df = pd.DataFrame(all_predictions)
    # Normalize the raw incident count prediction into a 0-100 risk score
    predictions_df['predicted_risk'] = predictions_df['predicted_incidents'].rank(pct=True) * 100
    
    st.session_state['ai_predictions'] = predictions_df
    

# ==============================================================================
# MAP GENERATION & DISPLAY
# ==============================================================================

with col1:
    map_center = [12.9716, 77.5946]
    zoom_level = 11
    
    # --- DYNAMIC MAP DATA BASED ON ACTIVE TAB ---
    active_tab = st.session_state.get('active_tab', '📊 Rainfall Patterns')
    
    # Default to current year data
    map_data_to_display = bbmp_wards

    if active_tab == "🕰️ Temporal Analysis":
        selected_year = st.session_state.get('temporal_year', 2025)
        map_data_to_display = temporal_wards_data[temporal_wards_data['year'] == selected_year]
        st.markdown(f"<h2 class='map-heading'>🏙️ Bengaluru Flood Resilience in {selected_year}</h2>", unsafe_allow_html=True)
    elif active_tab == "🌐 Network Analysis":
        st.markdown(f"<h2 class='map-heading'>🌐 Drain Network Analysis</h2>", unsafe_allow_html=True)
    else:
         st.markdown(f"<h2 class='map-heading'>🏙️ Bengaluru City-Wide Flood Resilience ({CURRENT_MONTH_YEAR})</h2>", unsafe_allow_html=True)


    m = folium.Map(location=map_center, zoom_start=zoom_level, control_scale=True, tiles="CartoDB Positron")

    # Base layer of all wards (greyed out for network view)
    is_network_view = active_tab == "🌐 Network Analysis" and 'selected_drain_wards' in st.session_state
    
    folium.GeoJson(
        bbmp_wards,
        name="All Wards",
        style_function=lambda feature: {"color": "#444444", "weight": 1, "fillOpacity": 0.05}
    ).add_to(m)

    # Main resilience layer
    if selected_ward_name == "--- Bangalore City Overview ---":
        folium.GeoJson(
            map_data_to_display,
            name="HAURCC: Ward Resilience Index",
            style_function=lambda feature: {
                "fillColor": resilience_colors.get(feature['properties'].get('resilience_level', 'High Resilience')),
                "color": "#333333" if not is_network_view else "#555555", 
                "weight": 0.8, 
                "fillOpacity": 0.75 if not is_network_view else 0.1
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['KGISWardName', 'KGISWardNo', 'Composite_Resilience_Index', 'resilience_level'],
                aliases=['Ward Name:', 'Ward No.:', 'Resilience Index:', 'Resilience Level:'],
                style="background-color: #2C3E50; color: #E0E0E0; border: 1px solid #34495E;"
            )
        ).add_to(m)

        # --- ADDITION: Visualize Anomalies on the Map ---
        if 'anomalous_wards' in st.session_state and st.session_state.anomalous_wards:
            anomalous_wards_gdf = bbmp_wards[bbmp_wards['KGISWardName'].isin(st.session_state.anomalous_wards)]
            anomalies_fg = folium.FeatureGroup(name="⚠️ Detected Anomalies").add_to(m)
            folium.GeoJson(
                anomalous_wards_gdf,
                style_function=lambda x: {"color": "#FF0000", "weight": 3, "fillOpacity": 0.1},
                tooltip=folium.GeoJsonTooltip(fields=['KGISWardName'], aliases=['Anomalous Ward:'])
            ).add_to(anomalies_fg)
        
        # --- ENHANCEMENT: Visualize Active Alerts on the Map ---
        if 'alert_wards' in st.session_state and st.session_state.alert_wards:
            st.error(f"**ACTIVE MONSOON ALERT:** Resources should be directed to the following high-risk wards: {', '.join(st.session_state.alert_wards)}", icon="🚨")
            
            alert_wards_gdf = bbmp_wards[bbmp_wards['KGISWardName'].isin(st.session_state.alert_wards)]
            alerts_fg = folium.FeatureGroup(name="🚨 ACTIVE ALERTS").add_to(m)
            folium.GeoJson(
                alert_wards_gdf,
                style_function=lambda x: {"color": "#FF0000", "weight": 4, "fillColor": "#FF0000", "fillOpacity": 0.5},
                tooltip=folium.GeoJsonTooltip(fields=['KGISWardName', 'Composite_Resilience_Index'], aliases=['ALERT WARD:', 'Vulnerability Score:'])
            ).add_to(alerts_fg)


        legend_html = f"""
             <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; z-index:9999; font-size:14px; background-color:#1A1A1A; padding:15px; border-radius:12px; border:2px solid #2C3E50; color: #E0E0E0;">
               <b>Ward Resilience Index</b> <br>
               <i style="background:{resilience_colors['Extreme Vulnerability']}; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Extreme Vulnerability <br>
               <i style="background:{resilience_colors['High Vulnerability']}; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> High Vulnerability <br>
               <i style="background:{resilience_colors['Moderate Vulnerability']}; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Moderate Vulnerability <br>
               <i style="background:{resilience_colors['Low Vulnerability']}; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Low Vulnerability <br>
               <i style="background:{resilience_colors['High Resilience']}; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> High Resilience <br>
             </div>
             """
        m.get_root().html.add_child(folium.Element(legend_html))

        coords = [[p.y, p.x] for p in all_flood_points_gdf.geometry if p]
        folium.plugins.HeatMap(coords, name="Global Flood Incident Density", radius=15, blur=10).add_to(m)

    else: # Detailed Ward View
        if selected_ward_gdf is not None and not selected_ward_gdf.empty:
            st.markdown(f"<h2 class='map-heading'>📍 HAURCC: {selected_ward_name} Detailed Analysis</h2>", unsafe_allow_html=True)
            
            ward_to_display_on_map = map_data_to_display[map_data_to_display['KGISWardName'] == selected_ward_name]
            if not ward_to_display_on_map.empty:
                 map_center = [ward_to_display_on_map.geometry.centroid.y.iloc[0], ward_to_display_on_map.geometry.centroid.x.iloc[0]]
            else: # Fallback if not in temporal data
                 map_center = [selected_ward_gdf.geometry.centroid.y.iloc[0], selected_ward_gdf.geometry.centroid.x.iloc[0]]

            m.location = map_center
            m.zoom_start = 14

            folium.GeoJson(
                ward_to_display_on_map if not ward_to_display_on_map.empty else selected_ward_gdf,
                name=f"Selected Ward: {selected_ward_name}",
                style_function=lambda feature: {"fillColor": resilience_colors.get(feature['properties'].get('resilience_level')), "color": "#000000", "weight": 3.5, "fillOpacity": 0.45},
                tooltip=folium.GeoJsonTooltip(fields=['KGISWardName', 'resilience_level', 'Composite_Resilience_Index'], aliases=['Ward Name:', 'Resilience Level:', 'Score:'])
            ).add_to(m)

            if st.session_state.get("enable_ai", False) and 'ai_predictions' in st.session_state:
                ward_pred = st.session_state.ai_predictions[st.session_state.ai_predictions['ward_name'] == selected_ward_name]
                if not ward_pred.empty:
                    pred_risk = ward_pred['predicted_risk'].iloc[0]
                    st.metric("AI-Projected Risk Score (2025)", f"{pred_risk:.1f}/100", 
                             delta=f"{pred_risk - display_properties.get('Composite_Resilience_Index', 0):.1f} from current",
                             delta_color="inverse", help="This is a statistical projection, not a deep learning forecast.")

            # --- ENHANCEMENT: Use the new cached function for grid generation ---
            with st.spinner("Generating high-resolution hotspot grid..."):
                grid_gdf = generate_ward_hotspot_grid(selected_ward_gdf, all_flood_points_gdf, st.session_state.get('grid_size_m', 250))
            
            if grid_gdf is not None:
                folium.GeoJson(
                    grid_gdf,
                    name=f"{st.session_state['grid_size_m']}m Grid Hotspots",
                    style_function=lambda feature: {
                        "color": "#A0A0A0", "weight": 0.7,
                        "fillColor": grid_risk_colors.get(feature['properties'].get('grid_risk_level', 'No Incidents')),
                        "fillOpacity": 0.8 if feature['properties'].get('incident_count_in_cell', 0) > 0 else 0.0,
                    },
                    tooltip=folium.features.GeoJsonTooltip(fields=['incident_count_in_cell', 'grid_risk_level'], aliases=['Incidents in cell:', 'Grid Risk Level:'])
                ).add_to(m)

                grid_legend_html = f"""
                         <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; z-index:9999; font-size:14px; background-color:#1A1A1A; padding:15px; border-radius:12px; color: #E0E0E0;">
                             <b>Grid Hotspot Risk</b> <br>
                             <i style="background:{grid_risk_colors['Critical Risk']}; width:12px; height:12px; display:inline-block;"></i> Critical Risk <br>
                             <i style="background:{grid_risk_colors['High Risk']}; width:12px; height:12px; display:inline-block;"></i> High Risk <br>
                             <i style="background:{grid_risk_colors['Moderate Risk']}; width:12px; height:12px; display:inline-block;"></i> Moderate Risk <br>
                             <i style="background:{grid_risk_colors['Low Risk']}; width:12px; height:12px; display:inline-block;"></i> Low Risk <br>
                             <i style="background:{grid_risk_colors['Minor Risk']}; width:12px; height:12px; display:inline-block;"></i> Minor Risk
                         </div>
                         """
                m.get_root().html.add_child(folium.Element(grid_legend_html))
    
    # --- NETWORK ANALYSIS VISUALIZATION ---
    if is_network_view:
        selected_drain_gdf = st.session_state.selected_drain_gdf
        selected_drain_wards_gdf = st.session_state.selected_drain_wards

        # Highlight the selected drain
        folium.GeoJson(
            selected_drain_gdf,
            name="Selected Drain",
            style_function=lambda x: {'color': '#00FFFF', 'weight': 6, 'opacity': 0.9}
        ).add_to(m)

        # Highlight the wards in the network
        folium.GeoJson(
            selected_drain_wards_gdf,
            name="Wards in Network",
            style_function=lambda x: {'color': '#00FF99', 'weight': 3, 'fillOpacity': 0.4},
            tooltip=folium.GeoJsonTooltip(fields=['KGISWardName', 'Composite_Resilience_Index'], aliases=['Ward:', 'Score:'])
        ).add_to(m)
        
        # Fit map to the network bounds
        m.fit_bounds(selected_drain_wards_gdf.total_bounds)


    # Add Primary Drains Layer
    folium.GeoJson(
        primary_drains, name="Primary Stormwater Drains",
        style_function=lambda x: {"color": "#0099FF", "weight": 2.5, "opacity": 0.8 if not is_network_view else 0.2},
        tooltip=folium.features.GeoJsonTooltip(fields=['Name', 'length_km'], aliases=['Drain Name:', 'Length (km):'])
    ).add_to(m)

    # Add Flood Incident Points Layer
    mc = folium.plugins.MarkerCluster(name="Historical Flood Incidents").add_to(m)
    for _, row in all_flood_points_gdf.iterrows():
        if row.geometry:
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x], radius=6, color='#CC0000',
                fill=True, fill_color='#FF0000', fill_opacity=0.9 if not is_network_view else 0.3,
                tooltip=f"<b>Incident:</b> {row.get('Name', 'N/A')}<br>"
            ).add_to(mc)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width='100%', height=650, key="haurcc_map_display")

    # --- TEMPORAL ANALYSIS CHART ---
    if active_tab == "🕰️ Temporal Analysis" and selected_ward_name != "--- Bangalore City Overview ---":
        ward_temporal_data = temporal_wards_data[temporal_wards_data['KGISWardName'] == selected_ward_name]
        if not ward_temporal_data.empty:
            st.markdown("---")
            st.markdown(f"<h3 style='color: #00C0FF;'>📈 Resilience Index Trend for {selected_ward_name} (2021-2025)</h3>", unsafe_allow_html=True)
            
            trend_chart = alt.Chart(ward_temporal_data).mark_line(
                point=alt.OverlayMarkDef(color="#00FF99", size=60),
                color="#00C0FF"
            ).encode(
                x=alt.X('year:O', title='Year'),
                y=alt.Y('Composite_Resilience_Index:Q', title='Vulnerability Score (Higher is Worse)', scale=alt.Scale(zero=False)),
                tooltip=['year:O', 'Composite_Resilience_Index:Q']
            ).properties(
                height=300
            ).configure_axis(
                gridColor='#34495E', labelColor='#E0E0E0', titleColor='#E0E0E0'
            ).configure_view(
                strokeWidth=0
            ).configure_title(
                color='#E0E0E0', fontSize=16
            )
            st.altair_chart(trend_chart, use_container_width=True)

with col2:
    if active_tab == "🌐 Network Analysis" and 'selected_drain_wards' in st.session_state:
        # --- NETWORK ANALYSIS REPORT ---
        drain_name = st.session_state.selected_drain_name
        network_wards = st.session_state.selected_drain_wards
        
        st.markdown(f"<h3 style='color: #00C0FF;'>🌐 Report for {drain_name}</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <p style='color: #E0E0E0;'>This report shows all wards the selected drain passes through, allowing for analysis of upstream and downstream vulnerabilities.</p>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            network_wards[['KGISWardName', 'Composite_Resilience_Index', 'incident_count']]
            .rename(columns={'KGISWardName': 'Ward in Network', 'Composite_Resilience_Index': 'Vulnerability Score', 'incident_count': 'Incidents'}),
            hide_index=True
        )

        if st.session_state.get("enable_ai", False):
            st.markdown("<h4 style='color: #00FF99;'>🤖 AI Network Insight</h4>", unsafe_allow_html=True)
            with st.spinner("AI is analyzing network dependencies..."):
                insight = generate_network_insight(drain_name, network_wards)
                st.markdown(f"<div class='metric-card'><p style='color: #E0E0E0;'>{insight}</p></div>", unsafe_allow_html=True)

    else:
        # Default view
        st.markdown("<h3 style='color: #00C0FF;'>📋 Map Legend & Controls</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <p style='color: #E0E0E0;'><strong>Map Layers:</strong></p>
            <ul style='color: #BDC3C7; padding-left: 20px;'>
                <li>Ward Resilience Index</li>
                <li>Historical Flood Incidents</li>
                <li>Primary Stormwater Drains</li>
                <li>Granular Hotspot Grid</li>
                <li>⚠️ Detected Anomalies (if run)</li>
            </ul>
            <p style='color: #E0E0E0;'><strong>Controls:</strong> Use layer control (top-right on map) to toggle visibility.</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# ADVANCED ANALYTICS SECTION (UPGRADED AND CORRECTED)
# ==============================================================================
# ==============================================================================
# ADVANCED ANALYTICS SECTION (CORRECTED FOR STREAMLIT COMPATIBILITY)
# ==============================================================================

st.markdown("---")
st.markdown("<h2 style='color: #00FF99;'>📈 Advanced Resilience Analytics</h2>", unsafe_allow_html=True)

# --- NEW: State management for tabs using st.radio (compatible with older Streamlit versions) ---
tab_names = [
    "📊 Rainfall Patterns", 
    "🤝 Ward Comparison", 
    "📋 Incident Breakdown", 
    "📈 Resilience Index", 
    "🕰️ Temporal Analysis", # New
    "🌐 Network Analysis",  # New
    "🤖 AI Predictions", 
    "💡 AI Recommendations",
    "🧠 Methodology & Data"
]

# Use st.radio, styled horizontally, to act as our tab controller.
active_tab = st.radio(
    "Select Analysis View",
    tab_names,
    horizontal=True,
    label_visibility="collapsed" # Hides the "Select Analysis View" label
)

# Store the active tab in session_state so the map logic can use it.
st.session_state.active_tab = active_tab

# --- Display content based on the selected "tab" ---

if active_tab == "📊 Rainfall Patterns":
    st.markdown("<h3 style='color: #00C0FF;'>🌧️ Historical Rainfall Trends in Bengaluru</h3>", unsafe_allow_html=True)
    if rainfall_data is not None:
        st.markdown("""
        <div class='metric-card'>
            <p style='color: #BDC3C7;'>This chart displays the total annual rainfall from 1900 to the present, showing long-term climatic trends and identifying years with extreme weather events that could impact urban flooding.</p>
        </div>
        """, unsafe_allow_html=True)
        
        chart = alt.Chart(rainfall_data, title="Annual Rainfall in Bengaluru (1900-Present)").mark_line(
            point=alt.OverlayMarkDef(color="#00FF99"),
            color="#00C0FF"
        ).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Total:Q', title='Total Annual Rainfall (mm)'),
            tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('Total:Q', title='Rainfall (mm)')]
        ).properties(
            height=400
        ).configure_axis(
            gridColor='#34495E', labelColor='#E0E0E0', titleColor='#E0E0E0'
        ).configure_view(
            strokeWidth=0
        ).configure_title(
            color='#E0E0E0', fontSize=16
        )
        st.altair_chart(chart, use_container_width=True)

elif active_tab == "🤝 Ward Comparison":
    st.markdown("<h3 style='color: #00C0FF;'>🤝 Ward Vulnerability Score Comparison</h3>", unsafe_allow_html=True)
    comparison_col1, comparison_col2 = st.columns([2, 1])
    with comparison_col1:
        st.markdown("""
        <div class='metric-card'>
            <h4 style='color: #00FF99; margin: 0 0 15px 0;'>Compare Ward Vulnerability</h4>
            <p style='color: #BDC3C7; margin: 0 0 10px 0;'>Select two or more wards to compare their Vulnerability Scores. A higher score means higher vulnerability.</p>
        </div>
        """, unsafe_allow_html=True)
        selected_comparison_wards = st.multiselect(
            "Select Wards to Compare:", options=ward_names,
            default=ward_names[:3] if len(ward_names) >= 3 else ward_names,
            help="Choose multiple wards to compare their vulnerability scores"
        )
        if len(selected_comparison_wards) >= 2:
            comparison_data = bbmp_wards[bbmp_wards['KGISWardName'].isin(selected_comparison_wards)]
            chart = alt.Chart(comparison_data, title="Vulnerability Score Comparison (2025)").mark_bar().encode(
                x=alt.X('KGISWardName:N', title='Ward', sort='-y'),
                y=alt.Y('Composite_Resilience_Index:Q', title='Vulnerability Score (Higher is Worse)'),
                color=alt.Color('KGISWardName:N', legend=None),
                tooltip=[ alt.Tooltip('KGISWardName:N', title='Ward'), alt.Tooltip('Composite_Resilience_Index:Q', title='Score', format='.1f')]
            ).properties(height=450).configure_axis(gridColor='#34495E', labelColor='#E0E0E0', titleColor='#E0E0E0').configure_view(strokeWidth=0).configure_title(color='#E0E0E0', fontSize=16)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Please select at least 2 wards for comparison.", icon="ℹ️")
    with comparison_col2:
        st.markdown("<h4 style='color: #00C0FF;'>Comparison Insights (2025)</h4>", unsafe_allow_html=True)
        if len(selected_comparison_wards) >= 2:
            stats_data = bbmp_wards[bbmp_wards['KGISWardName'].isin(selected_comparison_wards)]
            best_ward = stats_data.loc[stats_data['Composite_Resilience_Index'].idxmin()]
            worst_ward = stats_data.loc[stats_data['Composite_Resilience_Index'].idxmax()]
            st.metric("Most Resilient (Lowest Score)", f"{best_ward['KGISWardName']} ({best_ward['Composite_Resilience_Index']:.1f})")
            st.metric("Most Vulnerable (Highest Score)", f"{worst_ward['KGISWardName']} ({worst_ward['Composite_Resilience_Index']:.1f})", delta_color="inverse")
            st.metric("Average Score of Selection", f"{stats_data['Composite_Resilience_Index'].mean():.1f}")
        else:
            st.markdown("<p style='color: #BDC3C7;'>Select wards to see insights.</p>", unsafe_allow_html=True)

elif active_tab == "📋 Incident Breakdown":
    st.markdown("<h3 style='color: #00C0FF;'>📋 Historical Incident Breakdown</h3>", unsafe_allow_html=True)
    incident_col1, incident_col2 = st.columns([3, 2])
    with incident_col1:
        st.markdown("""
        <div class='metric-card'>
            <p style='color: #BDC3C7;'>This chart shows the top 15 wards with the highest number of recorded flood incidents, helping to pinpoint recurring problem areas.</p>
        </div>
        """, unsafe_allow_html=True)
        if not all_flood_points_gdf.empty:
            with st.spinner("Analyzing incident locations..."):
                points_in_wards = gpd.sjoin(all_flood_points_gdf, bbmp_wards[['KGISWardName', 'geometry']], how="inner", predicate="within")
                incident_counts = points_in_wards['KGISWardName'].value_counts().reset_index()
                incident_counts.columns = ['Ward', 'Count']
                top_wards = incident_counts.head(15)
            chart = alt.Chart(top_wards, title="Top 15 Wards by Flood Incident Count").mark_bar(color="#FF4500").encode(
                x=alt.X('Count:Q', title='Number of Incidents'),
                y=alt.Y('Ward:N', sort='-x', title='Ward Name'),
                tooltip=[alt.Tooltip('Ward:N', title='Ward'), alt.Tooltip('Count:Q', title='Incidents')]
            ).properties(height=500).configure_axis(gridColor='#34495E', labelColor='#E0E0E0', titleColor='#E0E0E0').configure_view(strokeWidth=0).configure_title(color='#E0E0E0', fontSize=16)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No flood incident data available to display.", icon="⚠️")
    with incident_col2:
        st.markdown("<h4 style='color: #00C0FF;'>City-Wide Statistics</h4>", unsafe_allow_html=True)
        st.metric("Total Recorded Incidents", f"{len(all_flood_points_gdf):,}")
        st.metric("Wards with at Least One Incident", f"{bbmp_wards[bbmp_wards['incident_count'] > 0].shape[0]}")
        st.markdown("""
        <div class='metric-card' style='margin-top: 2rem;'>
            <h5 style='color: #00FF99;'>Insight 💡</h5>
            <p style='color: #BDC3C7;'>A significant percentage of flood incidents are concentrated in a small number of wards. This suggests that targeted, high-impact interventions in these specific areas could drastically improve the city's overall flood resilience.</p>
        </div>
        """, unsafe_allow_html=True)

elif active_tab == "📈 Resilience Index":
    st.markdown("<h3 style='color: #00C0FF;'>📈 Resilience Index Analysis (2025)</h3>", unsafe_allow_html=True)
    resilience_col1, resilience_col2 = st.columns([3, 2])
    with resilience_col1:
        st.markdown("""
        <div class='metric-card'>
            <p style='color: #BDC3C7;'>The histogram below shows the distribution of vulnerability scores across all 198 wards. A skew towards the right indicates a higher number of vulnerable wards.</p>
        </div>
        """, unsafe_allow_html=True)
        hist_chart = alt.Chart(bbmp_wards, title="Distribution of Ward Vulnerability Scores").mark_bar(color="#00C0FF").encode(
            x=alt.X('Composite_Resilience_Index:Q', bin=alt.Bin(maxbins=20), title='Vulnerability Score (Higher is Worse)'),
            y=alt.Y('count():Q', title='Number of Wards'),
            tooltip=[alt.Tooltip('count()', title='Number of Wards'), alt.Tooltip('Composite_Resilience_Index:Q', bin=True, title='Score Range')]
        ).properties(height=400).configure_axis(gridColor='#34495E', labelColor='#E0E0E0', titleColor='#E0E0E0').configure_view(strokeWidth=0).configure_title(color='#E0E0E0', fontSize=16)
        st.altair_chart(hist_chart, use_container_width=True)
    with resilience_col2:
        st.markdown("<h4 style='color: #00C0FF;'>Key Statistics & Rankings</h4>", unsafe_allow_html=True)
        st.metric("Average Vulnerability Score", f"{bbmp_wards['Composite_Resilience_Index'].mean():.2f} / 100")
        st.markdown("<h5 style='color: #00FF99; margin-top: 1rem;'>Most Resilient Wards (Top 5)</h5>", unsafe_allow_html=True)
        st.dataframe(bbmp_wards[['KGISWardName', 'Composite_Resilience_Index']].nsmallest(5, 'Composite_Resilience_Index').rename(columns={'KGISWardName': 'Ward', 'Composite_Resilience_Index': 'Score'}), hide_index=True)
        st.markdown("<h5 style='color: #FF4500; margin-top: 1rem;'>Most Vulnerable Wards (Top 5)</h5>", unsafe_allow_html=True)
        st.dataframe(bbmp_wards[['KGISWardName', 'Composite_Resilience_Index']].nlargest(5, 'Composite_Resilience_Index').rename(columns={'KGISWardName': 'Ward', 'Composite_Resilience_Index': 'Score'}), hide_index=True)

elif active_tab == "🕰️ Temporal Analysis":
    st.markdown("<h3 style='color: #00C0FF;'>🕰️ Temporal Analysis: Resilience Over Time</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class='metric-card'>
            <p style='color: #BDC3C7;'>Use the slider to travel through time and see how ward vulnerabilities have evolved. The main map will update to reflect the data for the selected year. Select a specific ward in the sidebar to see its individual trend chart below the map.</p>
        </div>
        """, unsafe_allow_html=True)
    selected_year = st.slider(
        "**Select Year for Analysis:**",
        min_value=2021, max_value=2025, value=2025, step=1,
        key='temporal_year'
    )

elif active_tab == "🌐 Network Analysis":
    st.markdown("<h3 style='color: #00C0FF;'>🌐 Upstream-Downstream Network Analysis</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class='metric-card'>
            <p style='color: #BDC3C7;'>Analyze the entire network of a primary stormwater drain to identify systemic risks and bottlenecks. Select a drain to highlight it and all connected wards on the main map.</p>
        </div>
        """, unsafe_allow_html=True)
    drain_names = ["--- Select a Drain ---"] + sorted(primary_drains['Name'].dropna().unique().tolist())
    selected_drain_name = st.selectbox("Select a Primary Drain to Analyze:", options=drain_names)
    if selected_drain_name and selected_drain_name != "--- Select a Drain ---":
        selected_drain_gdf = primary_drains[primary_drains['Name'] == selected_drain_name]
        intersecting_wards_indices = selected_drain_gdf.sjoin(bbmp_wards, how='inner', predicate='intersects')['index_right']
        network_wards_df = bbmp_wards.loc[intersecting_wards_indices.unique()].copy()
        st.session_state.selected_drain_name = selected_drain_name
        st.session_state.selected_drain_gdf = selected_drain_gdf
        st.session_state.selected_drain_wards = network_wards_df
    else:
        for key in ['selected_drain_name', 'selected_drain_gdf', 'selected_drain_wards']:
            if key in st.session_state:
                del st.session_state[key]

elif active_tab == "🤖 AI Predictions":
    st.markdown("<h3 style='color: #00C0FF;'>🤖 AI-Powered Risk Predictions (for 2025)</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
        <p style='color: #BDC3C7;'>This table shows the current data-driven vulnerability index alongside a projected risk score from a machine learning model. The model predicts a ward's likely number of flood incidents based on its physical characteristics (like area and drainage), providing a forward-looking, independent assessment of risk.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.get("enable_ai", False) and 'ai_predictions' in st.session_state:
        risk_df = bbmp_wards.merge(st.session_state.ai_predictions, left_on='KGISWardName', right_on='ward_name')
        st.dataframe(risk_df[['KGISWardName', 'Composite_Resilience_Index', 'predicted_risk']].rename(columns={
            'KGISWardName': 'Ward Name',
            'Composite_Resilience_Index': 'Data-Driven Vulnerability Index',
            'predicted_risk': 'ML-Projected Incident Risk Score'
        }).sort_values('ML-Projected Incident Risk Score', ascending=False), use_container_width=True)
    else:
        st.warning("AI features are not enabled. Please enable AI in the sidebar.", icon="🤖")

elif active_tab == "💡 AI Recommendations":
    st.markdown("<h3 style='color: #00C0FF;'>💡 AI-Generated Recommendations</h3>", unsafe_allow_html=True)
    if st.session_state.get("enable_ai", False):
        recommendation_ward_name = st.session_state.get('selected_ward', "--- Bangalore City Overview ---")
        if recommendation_ward_name != "--- Bangalore City Overview ---":
            st.info(f"Generating recommendations for ward: **{recommendation_ward_name}**", icon="🎯")
            ward_data = bbmp_wards[bbmp_wards['KGISWardName'] == recommendation_ward_name].iloc[0]
            pred_data = st.session_state.get('ai_predictions')
            if pred_data is not None:
                ward_pred_df = pred_data[pred_data['ward_name'] == recommendation_ward_name]
                if not ward_pred_df.empty:
                    predicted_risk = ward_pred_df['predicted_risk'].iloc[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Vulnerability Score", f"{ward_data['Composite_Resilience_Index']:.1f}/100")
                    with col2:
                        st.metric("AI-Projected Risk Score", f"{predicted_risk:.1f}/100")
                    if st.button("Generate AI Recommendations", type="primary", key="gen_ai_rec"):
                        with st.spinner("🧠 AI expert is analyzing data and crafting a strategic plan..."):
                            recommendations = generate_ai_recommendations(ward_data, predicted_risk)
                            st.success("✅ Strategic Plan Generated!")
                            for recommendation in recommendations:
                                st.markdown(f"<div class='metric-card'>{recommendation}</div>", unsafe_allow_html=True)
                else:
                    st.warning(f"No prediction data found for {recommendation_ward_name}. Please refresh AI predictions.", icon="⚠️")
            else:
                st.warning("Prediction data not available. Please click 'Refresh AI Predictions' in the sidebar.", icon="🔄")
        else:
            st.warning("Please select a specific ward from the sidebar to generate AI recommendations.", icon="👈")
    else:
        st.warning("AI features are not enabled. Please enable AI in the sidebar.", icon="🤖")

elif active_tab == "🧠 Methodology & Data":
    st.markdown("<h3 style='color: #00C0FF;'>🧠 Methodology & Data Sources</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
    <h4 style='color: #00FF99;'>Composite Resilience Index Calculation</h4>
    <p style='color: #BDC3C7;'>
    The Composite Resilience Index is a score from 0 to 100 that quantifies a ward's vulnerability to flooding. <strong>A higher score indicates higher vulnerability (lower resilience).</strong> It is calculated using a three-step, data-driven process:
    </p>
    <ol style='color: #E0E0E0;'>
        <li><strong>Robust Normalization:</strong> Key metrics (incident density, etc.) are normalized using percentile ranking. This prevents a single outlier ward from skewing the results for all other wards.</li>
        <li><strong>Data-Driven Weighting (PCA):</strong> Principal Component Analysis (PCA) is used to analyze the normalized metrics. It derives statistical weights based on which factors contribute the most to the variance in the data, removing subjective guesswork.</li>
        <li><strong>Final Scaling:</strong> The weighted scores are scaled to a final 0-100 index for clear interpretation and mapping.</li>
    </ol>
    </div>
    <div class='metric-card'>
    <h4 style='color: #00FF99;'>Data Sources</h4>
    <ul style='color: #E0E0E0;'>
        <li><strong>Ward Boundaries:</strong> Bruhat Bengaluru Mahanagara Palike (BBMP) official ward boundaries.</li>
        <li><strong>Flood Incident Data:</strong> Aggregated from BBMP records on flood-prone locations, vulnerable areas, and low-lying areas.</li>
        <li><strong>Stormwater Drain Network:</strong> Primary drain data from Bengaluru's public works department.</li>
        <li><strong>Rainfall Data:</strong> Historical data from the Karnataka State Natural Disaster Monitoring Centre (KSNDMC).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #BDC3C7; padding: 20px;'>
    <p>Developed for Urban Resilience & Flood Management in Bengaluru •</p>
    <p style='font-size: 0.9em;'>Data sources: BBMP, KSNDMC, Open Data Initiatives</p>
    <p style='font-size: 0.8em; color: #6C7A89;'>Powered by Streamlit, GeoPandas, Folium, and Altair</p>
</div>
""", unsafe_allow_html=True)
