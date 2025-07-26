import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
import numpy as np
import pandas as pd
import requests
import altair as alt
from shapely.geometry import box, shape
import branca.colormap as cm

# --- Configuration ---
CURRENT_MONTH = "July 2025"
N8N_WEBHOOK_URL = "YOUR_N8N_WEBHOOK_URL_HERE" # <<< IMPORTANT: Replace with your n8n webhook URL

# --- AI Recommendation Rules (Simulated Dynamic Generation) ---
AI_RECOMMENDATIONS = {
    "Very High": [
        "**CRITICAL ACTION REQUIRED: Immediate Infrastructure Upgrade & Emergency Planning**",
        "- **Drainage Expansion:** Prioritize widening and deepening of primary and secondary drains in identified hotspots. For instance, sections along main roads showing persistent waterlogging require immediate hydraulic assessment for a target increase of 40-50% cross-section capacity.",
        "- **Siltation Removal:** Initiate emergency dredging and desilting operations, especially in drain segments leading to historical flood points. Focus on areas with observed high silt accumulation, targeting a 90% removal rate.",
        "- **Green Infrastructure:** Implement large-scale stormwater retention ponds and bioswales in open public spaces, designed to manage extreme rainfall events.",
        "- **Monitoring:** Deploy advanced IoT water level sensors at all critical choke points and low-lying areas for real-time flood monitoring and early warning dissemination to emergency services and residents.",
        "- **Evacuation Planning:** Refine and drill comprehensive, ward-specific evacuation routes and emergency shelter protocols. Conduct frequent community awareness programs."
    ],
    "High": [
        "**Urgent Action Needed: Capacity Enhancement & Proactive Measures**",
        "- **Drainage Upgrade:** Assess and upgrade current stormwater drainage capacity. Focus on structural improvements and repairing damaged sections. Consider increasing capacity by 20-30% in high-impact zones.",
        "- **Targeted Desilting:** Schedule priority desilting and clearing for drain segments known for frequent blockages or slow flow. Implement bi-monthly inspections for these segments.",
        "- **Permeable Pavement:** Identify suitable road sections or public pathways for conversion to permeable pavers to enhance local infiltration and reduce surface runoff by an estimated 10-15%.",
        "- **Rainwater Harvesting:** Promote and incentivize the installation of large-scale rainwater harvesting systems for commercial and large residential complexes.",
        "- **Community Preparedness:** Coordinate with local disaster management teams for rapid response and detailed evacuation plans. Conduct mock drills quarterly."
    ],
    "Moderate": [
        "**Strategic Planning & Maintenance: Mitigating Future Risks**",
        "- **Drainage Audits:** Conduct detailed localized drainage audits to identify choke points and minor blockages in the ward's storm drain network. Map out areas prone to silt accumulation for preventative action.",
        "- **Soak Pit Installation:** Identify 5-10 optimal locations for smart soak pit installation in available green spaces or unpaved shoulders to manage runoff from adjacent areas.",
        "- **Rain Garden Implementation:** Propose and design community rain gardens in parks or public green spaces to absorb stormwater from surrounding rooftops and impervious surfaces.",
        "- **Waste Management:** Intensify public awareness campaigns on proper waste disposal to prevent drain clogging. Increase frequency of waste collection in vulnerable areas.",
        "- **Regular Inspections:** Implement a scheduled bi-annual inspection program for drain structural integrity and minor repairs."
    ],
    "Low": [
        "**Preventative Maintenance & Long-Term Resilience Building**",
        "- **Routine Drain Cleaning:** Maintain routine cleaning and inspection of all primary and secondary drainage infrastructure. Ensure quarterly cleaning of culverts and smaller drains.",
        "- **Tree Cover Enhancement:** Support initiatives to increase urban tree cover, especially along roads and open spaces, to enhance natural water absorption and reduce heat island effect.",
        "- **Sustainable Land Use:** Review current land-use planning guidelines to ensure new developments incorporate robust stormwater management and green infrastructure principles.",
        "- **Public Education:** Continue educating residents on property-level flood protection measures and the importance of responsible water usage.",
        "- **Climate Monitoring:** Monitor long-term climate change predictions to anticipate future shifts in rainfall intensity and adapt strategies proactively."
    ],
    "No Incidents": [
        "**Proactive Urban Planning & Sustained Resilience**",
        "- **Infrastructure Maintenance:** Continue diligent maintenance of existing drainage infrastructure to prevent future vulnerabilities.",
        "- **Green Space Preservation:** Prioritize preservation and expansion of green spaces to enhance natural drainage and biodiversity.",
        "- **Smart Growth:** Ensure all new urban developments incorporate advanced stormwater management, permeable surfaces, and sustainable design principles.",
        "- **Community Engagement:** Foster community engagement in local environmental initiatives and emergency preparedness planning.",
        "- **Data Monitoring:** Continuously monitor environmental data, including rainfall patterns and land use changes, to adapt strategies as needed."
    ]
}

# --- Load and Process Data ---
@st.cache_data # Cache data to prevent reloading on every rerun
def load_and_process_data():
    try:
        # Load BBMP Wards (Polygons)
        wards_path = os.path.join("data", "bbmp-wards.geojson")
        wards_gdf = gpd.read_file(wards_path)
        if wards_gdf.crs is None or wards_gdf.crs.is_projected:
            wards_gdf = wards_gdf.to_crs("EPSG:4326")

        # Load Flood Incident Points from all three sources
        floodprone_path = os.path.join("data", "bbmp_floodprone_locations.geojson")
        vulnerable_path = os.path.join("data", "flooding_vulnerable_locations.geojson")
        lowlying_path = os.path.join("data", "bbmp_lowlying_areas.geojson")

        floodprone_gdf = gpd.read_file(floodprone_path)
        vulnerable_gdf = gpd.read_file(vulnerable_path)
        lowlying_gdf = gpd.read_file(lowlying_path)

        # Ensure all point GDFs have consistent CRS with wards GDF
        if floodprone_gdf.crs is None or floodprone_gdf.crs.is_projected:
            floodprone_gdf = floodprone_gdf.to_crs("EPSG:4326")
        if vulnerable_gdf.crs is None or vulnerable_gdf.crs.is_projected:
            vulnerable_gdf = vulnerable_gdf.to_crs("EPSG:4326")
        if lowlying_gdf.crs is None or lowlying_gdf.crs.is_projected:
            lowlying_gdf = lowlying_gdf.to_crs("EPSG:4326")

        # Combine all flood incident points into one GeoDataFrame
        all_flood_points_gdf = pd.concat([
            floodprone_gdf,
            vulnerable_gdf,
            lowlying_gdf
        ], ignore_index=True)
        
        # --- METRIC CALCULATION: Combined Proximity Flood Risk Score ---

        # 1. Calculate direct incident count (points within ward)
        wards_with_points = gpd.sjoin(all_flood_points_gdf, wards_gdf, how="inner", predicate="within")
        incident_counts = wards_with_points.groupby('index_right').size().rename("incident_count")
        wards_gdf = wards_gdf.merge(incident_counts, left_index=True, right_index=True, how="left")
        wards_gdf['incident_count'] = wards_gdf['incident_count'].fillna(0).astype(int)

        # 2. Calculate proximity incident count (points within a buffer around the ward)
        wards_gdf_proj = wards_gdf.to_crs(epsg=32643) # Project to UTM Zone 43N for Bengaluru for accurate buffering
        all_flood_points_gdf_proj = all_flood_points_gdf.to_crs(epsg=32643)

        buffered_wards_gdf_proj = wards_gdf_proj.copy()
        buffered_wards_gdf_proj['geometry'] = buffered_wards_gdf_proj.geometry.buffer(500) # 500 meters buffer

        wards_with_buffered_points = gpd.sjoin(all_flood_points_gdf_proj, buffered_wards_gdf_proj, how="inner", predicate="within")
        
        buffered_incident_counts = wards_with_buffered_points.groupby('index_right').size().rename("buffered_incident_count")

        wards_gdf = wards_gdf.merge(buffered_incident_counts, left_index=True, right_index=True, how="left")
        wards_gdf['buffered_incident_count'] = wards_gdf['buffered_incident_count'].fillna(0).astype(int)

        # 3. Create a combined risk score
        WARDS_WEIGHT = 2
        BUFFER_WEIGHT = 0.5 
        wards_gdf['combined_risk_score'] = (wards_gdf['incident_count'] * WARDS_WEIGHT) + (wards_gdf['buffered_incident_count'] * BUFFER_WEIGHT)
        
        # --- Assign Risk Level based on 'combined_risk_score' ---
        if not wards_gdf['combined_risk_score'].empty:
            def assign_risk_level_new_metric(score):
                if score >= 30: 
                    return "Very High"
                elif score >= 15: 
                    return "High"
                elif score >= 5: 
                    return "Moderate"
                elif score >= 1: 
                    return "Low"
                else: 
                    return "No Incidents"

            wards_gdf['risk_level'] = wards_gdf['combined_risk_score'].apply(assign_risk_level_new_metric)
        else:
            wards_gdf['risk_level'] = "No Data"

        # --- Load Drains Data ---
        drains_path = os.path.join("data", "bangalore_swd_primary.geojson")
        drains_gdf = gpd.read_file(drains_path)
        if drains_gdf.crs is None or drains_gdf.crs.is_projected:
            drains_gdf = drains_gdf.to_crs("EPSG:4326")

        # --- Load Rainfall Data ---
        rainfall_csv_path = os.path.join("data", "bangalore-rainfall-data-1900-2024-sept.csv")
        rainfall_df = pd.read_csv(rainfall_csv_path)
        rainfall_df['Year'] = pd.to_numeric(rainfall_df['Year'], errors='coerce').fillna(0).astype(int)
        rainfall_df.dropna(subset=['Total'], inplace=True)

        return wards_gdf, drains_gdf, rainfall_df, all_flood_points_gdf

    except Exception as e:
        st.error(f"Error loading or processing files: {e}. Please ensure all required files exist in the 'data/' folder and are valid GeoJSON/CSV. Also check the console for more details.")
        return None, None, None, None

# Load all necessary data
bbmp_wards, primary_drains, rainfall_data, all_flood_points_gdf = load_and_process_data()

if bbmp_wards is None or primary_drains is None or rainfall_data is None or all_flood_points_gdf is None:
    st.stop()

# --- Streamlit App Layout & Theming ---
st.set_page_config(
    page_title="Urban Resilience Dashboard - Bengaluru Wards",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6; /* Light gray background */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2E4053; /* Darker blue-gray for headings */
    }
    .stSelectbox, .stTextInput, .stButton {
        border-radius: 0.5rem;
    }
    .stSelectbox > div > div {
        border-color: #AAB7B8; /* Subtle border for select box */
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    .css-1d391kg { /* Adjust sidebar width for better content display */
        width: 350px;
    }
    /* Card-like effect for sections */
    .st-emotion-cache-nahz7x div { /* This targets the container for some widgets */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        background-color: white;
    }
    .css-1eereed { /* Specific targeting for the main map container */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        background-color: white;
    }
    .stPlotlyChart { /* For Altair charts */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        background-color: white;
    }
    /* Specific adjustments for selectbox/multiselect to remove extra padding if present */
    .stSelectbox > div {
        padding: 0px !important;
    }
    /* Ensure the main content doesn't get double-boxed if Streamlit adds its own outer container */
    .block-container.st-emotion-cache-nahz7x {
        box-shadow: none;
        background-color: transparent;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Define colors for ward risk levels (overall heatmap) ---
risk_colors = {
    "Very High": "#8B0000", # Darkest Red
    "High": "#DC143C",      # Crimson Red
    "Moderate": "#FF8C00",  # Dark Orange
    "Low": "#66CDAA",       # Medium Aquamarine (lighter green)
    "No Incidents": "#AAB7B8", # A neutral gray for wards with zero incidents
    "No Data": "#DDDDDD"    # Even lighter grey for wards where data couldn't be processed
}

# --- Define colors for GRID risk levels (localized heatmap) ---
grid_risk_colors = {
    "Critical Risk": "#800000",   # Very dark, almost maroon red
    "High Risk": "#990000",       # Very deep, rich, intense red
    "Moderate Risk": "#CC0000",   # Strong, classic, darker red (for the probable 4-6 incidents range)
    "Low Risk": "#FF7F7F",        # Medium-light red
    "Minor Risk": "#FFCCCB",      # Soft, light red
    "No Incidents": "#00000000"   # Fully transparent
}

def assign_grid_risk_level(incident_count):
    if incident_count == 0:
        return "No Incidents"
    elif incident_count == 1:
        return "Minor Risk"
    elif incident_count <= 3: # 2 or 3 incidents
        return "Low Risk"
    elif incident_count <= 6: # 4 to 6 incidents
        return "Moderate Risk"
    elif incident_count <= 10: # 7 to 10 incidents
        return "High Risk"
    else: # 11+ incidents
        return "Critical Risk"


# --- Header Section ---
st.container()
st.title("🏙️ Urban Resilience Dashboard - Bengaluru Floods")
st.markdown(f"""
    <p style='font-size: 1.1em; color: #566573;'>
        Gain insights into flood risk across Bengaluru's BBMP Wards. 
        <br><b>Initial View:</b> See a city-wide heatmap of all wards, colored by their overall flood risk.
        <br><b>Localized View:</b> Select a specific ward from the dropdown to zoom in and explore detailed 250m grid-based hotspots.
    </p>
    """, unsafe_allow_html=True)
st.markdown("---") # Visual separator

# --- Sidebar for Ward Selection and Details ---
with st.sidebar:
    st.header("🔍 Ward Selection & Insights")
    st.markdown("Select an option below to view its corresponding flood risk map and detailed information.")

    # Dropdown for Ward Selection
    ward_names = sorted(bbmp_wards['KGISWardName'].dropna().unique().tolist())
    ward_options = ["--- Bangalore as a Whole ---"] + ward_names # Add the "Bangalore as a Whole" option

    selected_ward_name = st.selectbox(
        "Choose a View:",
        options=ward_options,
        key="ward_selector"
    )

    selected_ward_gdf = None
    display_properties = None

    if selected_ward_name != "--- Bangalore as a Whole ---":
        selected_ward_gdf = bbmp_wards[bbmp_wards['KGISWardName'] == selected_ward_name].copy()
        if not selected_ward_gdf.empty:
            display_properties = selected_ward_gdf.iloc[0].to_dict()
            st.markdown(f"### **🏡 {display_properties.get('KGISWardName', 'N/A')}**")
            st.markdown(f"**Ward No.:** {display_properties.get('KGISWardNo', 'N/A')}")
            st.markdown(f"**Calculated Risk Level:** <span style='color: {risk_colors.get(display_properties.get('risk_level', 'No Data'))}; font-weight: bold;'>{display_properties.get('risk_level', 'No Data')}</span>", unsafe_allow_html=True)
            
            # Explicit explanation of risk calculation
            st.markdown(f"""
                <p style='font-size: 0.9em; font-style: italic; color: #566573;'>
                <br>This overall ward risk is calculated based on:
                <ul>
                    <li><b>{display_properties.get('incident_count', 'N/A')}</b> direct flood incidents *within* this ward.</li>
                    <li><b>{display_properties.get('buffered_incident_count', 'N/A')}</b> incidents in *neighboring areas* (within a 500m buffer).</li>
                    <li>This combines into a <b>Proximity Flood Risk Score: {display_properties.get('combined_risk_score', 'N/A'):.2f}</b>.</li>
                </ul>
                </p>
                """, unsafe_allow_html=True)

            
            st.markdown("---")
            st.subheader("💡 AI-Generated Actionable Recommendations")
            
            if N8N_WEBHOOK_URL == "YOUR_N8N_WEBHOOK_URL_HERE":
                st.warning("*(Note: For dynamic AI recommendations, replace 'YOUR_N8N_WEBHOOK_URL_HERE' in `app.py` with your actual n8n webhook URL.)*")
                recommendations_to_display = AI_RECOMMENDATIONS.get(display_properties.get('risk_level', 'No Data'), [])
            else:
                payload = {
                    "ward_name": selected_ward_name,
                    "ward_no": display_properties.get('KGISWardNo', 'N/A'),
                    "risk_level": display_properties.get('risk_level', 'No Data'),
                    "direct_incidents": display_properties.get('incident_count', 'N/A'),
                    "nearby_incidents": display_properties.get('buffered_incident_count', 'N/A'),
                    "proximity_risk_score": display_properties.get('combined_risk_score', 'N/A')
                }
                
                try:
                    with st.spinner("Fetching AI recommendations from n8n..."):
                        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        n8n_response_data = response.json()
                        recommendations_to_display = n8n_response_data.get("recommendations", [])
                        if not recommendations_to_display:
                            st.info("n8n returned no specific recommendations. Using fallback.")
                            recommendations_to_display = AI_RECOMMENDATIONS.get(display_properties.get('risk_level', 'No Data'), [])
                    else:
                        st.error(f"Error fetching recommendations from n8n: Status Code {response.status_code}. Using fallback.")
                        recommendations_to_display = AI_RECOMMENDATIONS.get(display_properties.get('risk_level', 'No Data'), [])

                except requests.exceptions.RequestException as e:
                    st.error(f"Network error communicating with n8n: {e}. Using fallback recommendations.")
                    recommendations_to_display = AI_RECOMMENDATIONS.get(display_properties.get('risk_level', 'No Data'), [])

            for i, rec in enumerate(recommendations_to_display):
                st.write(f"- {rec}")
            
            st.markdown("---")
            st.subheader("📊 Local Grid Analysis (250m x 250m)")
            st.markdown("""
                <p style='font-size: 0.9em; font-style: italic; color: #566573;'>
                This 250m grid heatmap shows flood incidents <b>only within each specific grid cell</b>. 
                <br>This provides a granular view of localized hotspots, which may differ from the overall ward risk. Hover for details.
                </p>
                """, unsafe_allow_html=True)


        else:
            st.info("No data found for the selected ward. Please choose another ward.")
    else:
        st.info("Select a specific ward from the dropdown to unlock detailed insights and localized hotspot analysis.")

# --- Main Map Display Area ---
st.container()
if selected_ward_name == "--- Bangalore as a Whole ---":
    st.subheader("🗺️ Bengaluru City-Wide Flood Risk Overview")
    map_center = [12.9716, 77.5946] # Center of Bengaluru
    zoom_level = 11

    m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="OpenStreetMap")

    # Add ALL BBMP Wards, colored by their risk level
    folium.GeoJson(
        bbmp_wards.__geo_interface__,
        name="All BBMP Wards (Overall Risk)",
        style_function=lambda feature: {
            "fillColor": risk_colors.get(feature['properties'].get('risk_level', 'No Data'), "#DDDDDD"),
            "color": "black",
            "weight": 0.5, # Thinner border for overall map
            "fillOpacity": 0.7
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['KGISWardName', 'KGISWardNo', 'incident_count', 'combined_risk_score', 'risk_level'],
            aliases=['Ward Name:', 'Ward No.:', 'Direct Incidents:', 'Proximity Risk Score:', 'Calculated Risk:'],
            localize=True
        )
    ).add_to(m)

    # Add a custom legend for the overall risk levels
    legend_html = """
             <div style="position: fixed; 
                         bottom: 50px; left: 50px; width: 150px; height: 160px; 
                         border:2px solid grey; z-index:9999; font-size:14px;
                         background-color:white; opacity:0.9; padding:10px; border-radius:10px;">
                &nbsp; <b>Ward Risk Level</b> <br>
                &nbsp; <i style="background:{}"></i> Very High <br>
                &nbsp; <i style="background:{}"></i> High <br>
                &nbsp; <i style="background:{}"></i> Moderate <br>
                &nbsp; <i style="background:{}"></i> Low <br>
                &nbsp; <i style="background:{}"></i> No Incidents <br>
             </div>
             """.format(risk_colors["Very High"], risk_colors["High"], risk_colors["Moderate"], 
                         risk_colors["Low"], risk_colors["No Incidents"])

    m.get_root().html.add_child(folium.Element(legend_html))


    st.info("Currently viewing the city-wide flood risk heatmap. Select a ward from the sidebar for a detailed view with grid hotspots.")

else: # A specific ward is selected
    if selected_ward_gdf is not None and not selected_ward_gdf.empty:
        st.subheader(f"📍 {selected_ward_name} (Ward No. {selected_ward_gdf.iloc[0]['KGISWardNo']}) - Localized Flood Hotspots")
        
        # Adjust map center and zoom for the selected ward
        map_center = [selected_ward_gdf.geometry.centroid.y.iloc[0], selected_ward_gdf.geometry.centroid.x.iloc[0]]
        zoom_level = 14 

        m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="OpenStreetMap")

        # Add the SELECTED BBMP Ward boundary
        folium.GeoJson(
            selected_ward_gdf.__geo_interface__,
            name=f"Selected Ward: {selected_ward_name}",
            style_function=lambda feature: {
                "fillColor": risk_colors.get(feature['properties'].get('risk_level', 'No Data'), "#DDDDDD"), # Fill with overall risk color
                "color": "#000000", # Black border for the selected ward
                "weight": 3, 
                "fillOpacity": 0.4 # Less opaque to see grid through
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['KGISWardName', 'KGISWardNo', 'incident_count', 'buffered_incident_count', 'combined_risk_score', 'risk_level'],
                aliases=['Ward Name:', 'Ward No.:', 'Direct Incidents:', 'Nearby Incidents (500m):', 'Proximity Risk Score:', 'Calculated Risk:'],
                localize=True
            )
        ).add_to(m)

        # --- Grid Generation and Display for Selected Ward with Incident Counts ---
        try:
            clicked_shape = selected_ward_gdf.geometry.iloc[0] 
            clicked_gdf = gpd.GeoDataFrame([1], geometry=[clicked_shape], crs="EPSG:4326")
            clicked_gdf_proj = clicked_gdf.to_crs("EPSG:32643")
            
            minx, miny, maxx, maxy = clicked_gdf_proj.total_bounds
            grid_size_meters = 250
            polygons = []
            
            x_coords = np.arange(minx, maxx + grid_size_meters, grid_size_meters)
            y_coords = np.arange(miny, maxy + grid_size_meters, grid_size_meters)

            for i in range(len(x_coords) - 1):
                for j in range(len(y_coords) - 1):
                    x1, y1 = x_coords[i], y_coords[j]
                    x2, y2 = x_coords[i+1], y_coords[j+1]
                    grid_cell = box(x1, y1, x2, y2)
                    
                    if clicked_gdf_proj.geometry.iloc[0].intersects(grid_cell):
                        polygons.append(grid_cell)
            
            if polygons:
                grid_gdf_proj = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:32643")
                grid_gdf = grid_gdf_proj.to_crs("EPSG:4326")

                grid_with_points = gpd.sjoin(grid_gdf, all_flood_points_gdf, how="left", predicate="intersects")
                incident_counts_per_grid_cell = grid_with_points.groupby(grid_with_points.index).size().rename("incident_count_in_cell")
                grid_gdf = grid_gdf.merge(incident_counts_per_grid_cell, left_index=True, right_index=True, how="left")
                grid_gdf['incident_count_in_cell'] = grid_gdf['incident_count_in_cell'].fillna(0).astype(int)

                # Assign grid-specific risk level based on incident count
                grid_gdf['grid_risk_level'] = grid_gdf['incident_count_in_cell'].apply(assign_grid_risk_level)
                
                folium.GeoJson(
                    grid_gdf.__geo_interface__,
                    name=f"250m Grid for {selected_ward_name} (Hotspots)",
                    style_function=lambda feature: {
                        "color": "#FF0000",  # Bright red lines for grid
                        "weight": 2,         # Slightly thicker grid lines
                        "fillColor": grid_risk_colors.get(feature['properties'].get('grid_risk_level', 'No Incidents')), # Use distinct grid risk colors
                        "fillOpacity": 0.8 if feature['properties'].get('incident_count_in_cell', 0) > 0 else 0.0,   # Increased fill opacity for active cells, transparent for 0 incidents
                    },
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['incident_count_in_cell', 'grid_risk_level'],
                        aliases=['Flood Incidents in this cell:', 'Grid Risk Level:'],
                        localize=True
                    )
                ).add_to(m)
                st.info("Currently viewing localized flood hotspots with grid. Hover over grid cells for incident counts and grid risk level.")

                # Add a custom legend for the GRID risk levels
                grid_legend_html = """
                     <div style="position: fixed; 
                                 bottom: 50px; left: 50px; width: 150px; height: 190px; 
                                 border:2px solid grey; z-index:9999; font-size:14px;
                                 background-color:white; opacity:0.9; padding:10px; border-radius:10px;">
                        &nbsp; <b>Grid Hotspot Risk</b> <br>
                        &nbsp; <i style="background:{}"></i> Critical Risk <br>
                        &nbsp; <i style="background:{}"></i> High Risk <br>
                        &nbsp; <i style="background:{}"></i> Moderate Risk <br>
                        &nbsp; <i style="background:{}"></i> Low Risk <br>
                        &nbsp; <i style="background:{}"></i> Minor Risk <br>
                        &nbsp; <i style="background:{}"></i> No Incidents <br>
                     </div>
                     """.format(grid_risk_colors["Critical Risk"], grid_risk_colors["High Risk"], grid_risk_colors["Moderate Risk"], 
                                 grid_risk_colors["Low Risk"], grid_risk_colors["Minor Risk"], grid_risk_colors["No Incidents"])

                m.get_root().html.add_child(folium.Element(grid_legend_html))


            else:
                st.warning(f"Could not generate any intersecting 250m grid cells for {selected_ward_name}. It might be too small or have an unusual shape.")

        except Exception as e:
            st.error(f"Error during grid generation for {selected_ward_name}: {e}")
            st.exception(e)

    else:
        st.warning(f"No GeoData found for ward: {selected_ward_name}. Please check the ward name in your GeoJSON data.")

# Add Primary Stormwater Drains Layer (always available to toggle)
if not primary_drains.empty:
    folium.GeoJson(
        primary_drains.__geo_interface__,
        name="Primary Stormwater Drains", # Layer name for control
        style_function=lambda x: {
            "color": "#0000FF",  # Blue color for drains
            "weight": 2,         # Line thickness
            "opacity": 0.7       # Opacity
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['Name', 'Description'], # Adjust fields based on your drains_gdf properties
            aliases=['Drain Name:', 'Description:'],
            localize=True
        )
    ).add_to(m)

# Add ALL Flood Incident Points (Historical Markers)
if not all_flood_points_gdf.empty:
    # Cluster markers for better visualization if there are many points
    mc = folium.plugins.MarkerCluster().add_to(m)
    for idx, row in all_flood_points_gdf.iterrows():
        # Ensure geometry is a Point
        if row.geometry.geom_type == 'Point':
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5, # Small circle marker
                color='red',
                fill=True,
                fill_color='darkred',
                fill_opacity=0.8,
                tooltip=f"Incident: {row.get('Name', 'N/A')}<br>Type: {row.get('Category', 'N/A')}" # Adjust fields based on your flood points GDF
            ).add_to(mc) # Add to marker cluster

# Add Layer Control to toggle layers
folium.LayerControl().add_to(m)

# Display the map in Streamlit
st_folium(m, width='100%', height=500, key="ward_map_display", returned_objects=[])

# --- Rainfall Data Visualization ---
st.markdown("---") 
st.container()
st.header("💧 Historical Rainfall Trends in Bengaluru")
st.markdown("<p style='font-size: 1em; color: #566573;'>Understanding historical rainfall patterns is crucial for urban flood resilience planning.</p>", unsafe_allow_html=True)

if rainfall_data is not None and not rainfall_data.empty:
    annual_rainfall_chart = alt.Chart(rainfall_data).mark_line(point=True).encode(
        x=alt.X('Year:O', axis=alt.Axis(format='d', title='Year')),
        y=alt.Y('Total:Q', title='Total Annual Rainfall (mm)'),
        tooltip=['Year', 'Total']
    ).properties(
        title='Total Annual Rainfall (1901-2024)'
    ).interactive()
    st.altair_chart(annual_rainfall_chart, use_container_width=True)

    st.markdown("---")

    monthly_rainfall_df = rainfall_data.melt(
        id_vars=['Year', 'Total', 'El NiNo (Y/N)', 'La Nina (Y/N)'],
        var_name='Month',
        value_name='Rainfall'
    )
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    monthly_rainfall_df = monthly_rainfall_df[monthly_rainfall_df['Month'].isin(months_order)]

    average_monthly_rainfall = monthly_rainfall_df.groupby('Month')['Rainfall'].mean().reset_index()
    average_monthly_rainfall['Month'] = pd.Categorical(average_monthly_rainfall['Month'], categories=months_order, ordered=True)
    average_monthly_rainfall = average_monthly_rainfall.sort_values('Month')

    # CORRECTED LINE: Used average_monthly_rainfall as data source
    monthly_rainfall_chart = alt.Chart(average_monthly_rainfall).mark_bar().encode(
        x=alt.X('Month:O', sort=months_order, title='Month'),
        y=alt.Y('Rainfall:Q', title='Average Rainfall (mm)'),
        tooltip=['Month', alt.Tooltip('Rainfall', format='.1f', title='Avg. Rainfall')]
    ).properties(
        title='Average Monthly Rainfall Pattern'
    )
    st.altair_chart(monthly_rainfall_chart, use_container_width=True)

    st.markdown("---")

    st.subheader("🌐 El Niño and La Niña Years Impact")
    st.write("These global climate phenomena can significantly influence local rainfall patterns.")
    
    el_nino_years = rainfall_data[rainfall_data['El NiNo (Y/N)'] == 'Y']['Year'].tolist()
    la_nina_years = rainfall_data[rainfall_data['La Nina (Y/N)'] == 'Y']['Year'].tolist()

    if el_nino_years:
        st.write(f"**El Niño Years:** {', '.join(map(str, el_nino_years))}")
    if la_nina_years:
        st.write(f"**La Niña Years:** {', '.join(map(str, la_nina_years))}")
    if not el_nino_years and not la_nina_years:
        st.info("No El Niño or La Niña years identified in the dataset.")

else:
    st.warning("Rainfall data could not be loaded or is empty. Please check the 'bangalore-rainfall-data-1900-2024-sept.csv' file.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Built for Urban Resilience Hackathon Demo - July 2025</p>", unsafe_allow_html=True)