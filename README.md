HAURCC — Hyper-Analytical Urban Resilience Command Center (v2.0)

An interactive Streamlit application for urban flood resilience analysis in Bengaluru.

📌 Overview

HAURCC provides geospatial insights and analytical dashboards to help urban planners, disaster management teams, and policymakers in flood resilience planning. Built with Python and Streamlit, it leverages historical and civic datasets to visualize vulnerabilities and identify high-risk zones.

🚀 Features

City-Wide Resilience Overview — Multi-factor resilience index for all BBMP wards, color-coded from High Resilience to Extreme Vulnerability.

Dynamic Geospatial Analysis — Ward-level hotspot grid analysis to identify micro-zones at risk.

Analytics Dashboard with tabs for:

Rainfall Patterns — Trends and anomalies.

Ward Comparison — Compare resilience metrics for up to 5 wards.

Incident Breakdown — Historical flood incident distribution.

Interactive Map Layers — Toggle stormwater drains, incident markers, low-lying areas, and basemaps.

🛠 Technology Stack

Frontend: StreamlitGeospatial Processing: GeoPandas, ShapelyMapping: Folium, streamlit-folium, BrancaVisualization: AltairData Handling: Pandas, NumPy

📂 Project Structure

haurcc/
├─ app.py
├─ requirements.txt
├─ data/
│  ├─ bbmp-wards.geojson
│  ├─ bangalore_swd_primary.geojson
│  ├─ bbmp_floodprone_locations.geojson
│  ├─ flooding_vulnerable_locations.geojson
│  ├─ bbmp_lowlying_areas.geojson
│  └─ bangalore-rainfall-data-1900-2024-sept.csv
└─ README.md

⚙️ Installation

git clone <repository-url>
cd haurcc
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

requirements.txt

streamlit
geopandas
folium
streamlit-folium
numpy
pandas
altair
shapely
branca

📊 Data Requirements

Place in data/:

bbmp-wards.geojson

bangalore_swd_primary.geojson

bbmp_floodprone_locations.geojson

flooding_vulnerable_locations.geojson

bbmp_lowlying_areas.geojson

bangalore-rainfall-data-1900-2024-sept.csv

▶️ Running the App

streamlit run app.py

Opens in your default browser.

⚠️ Disclaimer

HAURCC uses static datasets and is for demonstration only. Do not use for real-time operational decisions.

🤝 Contributing

Pull requests and issues are welcome. Suggestions:

Improve resilience index methodology.

Add CRS auto-detection and reprojection.

Optimize large GeoJSON handling.

📜 License

Specify license here (e.g., MIT).

Version: v2.0
