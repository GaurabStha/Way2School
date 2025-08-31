# Way2School

Way2School is a data-driven school transport planning tool developed for GovHack ACT 2025.  
It leverages government open datasets and graph analytics to make school journeys faster, safer, and more flexible for families.

🚸 Key Features:
- Predicts Park & Ride availability and risk before departure.
- Provides multiple multimodal routes (drive, bus, light rail, walk, bike) with ETA, fare, and safety score.
- Identifies schools and stations lacking Park & Ride within 600m, suggesting alternatives within 2km.
- Uses OpenStreetMap pedestrian networks to generate real walking paths for the last mile.
- Simulates live trip tracking with notifications for parents.
- Helps planners identify where new Park & Ride or alternative parking sites will have the most impact.

📊 Data Sources:
- ACT School Bus Services  
- Park & Ride Locations  
- Census Data for ACT Schools  
- Students by SA1  
- Bus Routes (MULTILINESTRING)  
- OpenStreetMap pedestrian networks  

🌱 Impact:
Way2School diverts cars from school gates during peak windows, improves child safety on the last mile, and equips planners with evidence for smarter infrastructure investment.

Check out [Way2School](https://way2school.streamlit.app/).

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/GaurabStha/Way2School.git
cd Way2School
```
Install Dependencies:
```bash
pip install -r requirements.txt
```
To Run Main Python File:
```bash
streamlit run app.py
```

To Run mvp_plus_routes.py file:
```bash
python mvp_plus_routes.py \
  --park_ride "datasets/Park_And_Ride_Locations.csv" \
  --school_buses "datasets/ACT_School_Bus_Services.csv" \
  --school_census "datasets/Census_Data_for_all_ACT_Schools_20250830.csv" \
  --students_distance "datasets/Students_Distance_from_School_-_by_school_and_SA1_20250830.csv" \
  --out_csv out/recommended_candidates.csv \
  --out_map out/recommendations_map.html \
  --time 08:15 \
  --bus_routes "datasets/Bus_Routes.csv" \
  --osm_walk \
  --osm_dist 2500
```
