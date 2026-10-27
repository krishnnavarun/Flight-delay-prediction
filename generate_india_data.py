"""
Script to generate realistic, comprehensive Indian domestic flight delay dataset
with heavy focus on Tamil Nadu airports (Coimbatore CJB, Chennai MAA, Madurai IXM, 
Tiruchirappalli TRZ, Tuticorin TCR, Salem SXV) and major connecting metros 
(Bengaluru BLR, Mumbai BOM, Delhi DEL, Hyderabad HYD, Kochi COK, Goa GOI, Pune PNQ, Kolkata CCU).

Authentic attributes based on DGCA (Directorate General of Civil Aviation, India) standards:
- 15-minute delay threshold for 'delayed' indicator
- Realistic route distances, scheduled flight durations
- Tamil Nadu seasonal factors: North-East Monsoon (Oct-Dec) in Chennai & coastal TN,
  South-West Monsoon (Jun-Aug), Winter Fog in North-South connecting routes (Dec-Jan)
- Airline carriers: IndiGo (6E), Air India (AI), SpiceJet (SG), Akasa Air (QP), Vistara (UK), Alliance Air (9I)
- Realistic delay causes: Weather, ATC Congestion, Late Inbound Aircraft, Airline Operations
"""

import os
import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

# 1. Airports Master
AIRPORTS = {
    'CJB': {'name': 'Coimbatore International Airport', 'city': 'Coimbatore', 'state': 'Tamil Nadu', 'type': 'International', 'lat': 11.0298, 'lon': 77.0434},
    'MAA': {'name': 'Chennai International Airport', 'city': 'Chennai', 'state': 'Tamil Nadu', 'type': 'Major Hub', 'lat': 12.9941, 'lon': 80.1709},
    'IXM': {'name': 'Madurai Airport', 'city': 'Madurai', 'state': 'Tamil Nadu', 'type': 'Customs/Domestic', 'lat': 9.8345, 'lon': 78.0934},
    'TRZ': {'name': 'Tiruchirappalli International Airport', 'city': 'Tiruchirappalli', 'state': 'Tamil Nadu', 'type': 'International', 'lat': 10.7654, 'lon': 78.7097},
    'TCR': {'name': 'Tuticorin Airport', 'city': 'Thoothukudi', 'state': 'Tamil Nadu', 'type': 'Domestic', 'lat': 8.7239, 'lon': 78.0264},
    'SXV': {'name': 'Salem Airport', 'city': 'Salem', 'state': 'Tamil Nadu', 'type': 'Domestic', 'lat': 11.7833, 'lon': 78.0650},
    'BLR': {'name': 'Kempegowda International Airport', 'city': 'Bengaluru', 'state': 'Karnataka', 'type': 'Major Hub', 'lat': 13.1986, 'lon': 77.7066},
    'BOM': {'name': 'Chhatrapati Shivaji Maharaj International Airport', 'city': 'Mumbai', 'state': 'Maharashtra', 'type': 'Major Hub', 'lat': 19.0896, 'lon': 72.8656},
    'DEL': {'name': 'Indira Gandhi International Airport', 'city': 'New Delhi', 'state': 'Delhi NCR', 'type': 'Major Hub', 'lat': 28.5562, 'lon': 77.1000},
    'HYD': {'name': 'Rajiv Gandhi International Airport', 'city': 'Hyderabad', 'state': 'Telangana', 'type': 'Major Hub', 'lat': 17.2403, 'lon': 78.4294},
    'COK': {'name': 'Cochin International Airport', 'city': 'Kochi', 'state': 'Kerala', 'type': 'International', 'lat': 10.1518, 'lon': 76.3930},
    'GOI': {'name': 'Dabolim / Manohar International Airport', 'city': 'Goa', 'state': 'Goa', 'type': 'Tourist Hub', 'lat': 15.3808, 'lon': 73.8313},
    'PNQ': {'name': 'Pune International Airport', 'city': 'Pune', 'state': 'Maharashtra', 'type': 'Commercial Hub', 'lat': 18.5822, 'lon': 73.9197},
    'CCU': {'name': 'Netaji Subhash Chandra Bose International Airport', 'city': 'Kolkata', 'state': 'West Bengal', 'type': 'Major Hub', 'lat': 22.6547, 'lon': 88.4467}
}

# 2. Carriers Master
CARRIERS = {
    '6E': {'name': 'IndiGo Airlines', 'market_share': 0.62, 'base_punctuality': 0.85},
    'AI': {'name': 'Air India', 'market_share': 0.14, 'base_punctuality': 0.76},
    'UK': {'name': 'Vistara', 'market_share': 0.09, 'base_punctuality': 0.84},
    'QP': {'name': 'Akasa Air', 'market_share': 0.05, 'base_punctuality': 0.81},
    'SG': {'name': 'SpiceJet', 'market_share': 0.06, 'base_punctuality': 0.68},
    '9I': {'name': 'Alliance Air (Regional/UDAN)', 'market_share': 0.04, 'base_punctuality': 0.72}
}

# 3. Route Network with standard distances (km) and air times (mins)
ROUTES = [
    # Tamil Nadu inter-city & feeder routes
    ('CJB', 'MAA', 400, 55),
    ('MAA', 'CJB', 400, 55),
    ('IXM', 'MAA', 420, 60),
    ('MAA', 'IXM', 420, 60),
    ('TRZ', 'MAA', 320, 50),
    ('MAA', 'TRZ', 320, 50),
    ('TCR', 'MAA', 520, 75),
    ('MAA', 'TCR', 520, 75),
    ('SXV', 'MAA', 280, 45),
    ('MAA', 'SXV', 280, 45),
    ('CJB', 'IXM', 180, 35),
    ('IXM', 'CJB', 180, 35),

    # Tamil Nadu to Bengaluru
    ('CJB', 'BLR', 260, 45),
    ('BLR', 'CJB', 260, 45),
    ('MAA', 'BLR', 290, 50),
    ('BLR', 'MAA', 290, 50),
    ('IXM', 'BLR', 350, 55),
    ('BLR', 'IXM', 350, 55),
    ('TRZ', 'BLR', 330, 50),
    ('BLR', 'TRZ', 330, 50),

    # Tamil Nadu to Mumbai
    ('CJB', 'BOM', 990, 105),
    ('BOM', 'CJB', 990, 105),
    ('MAA', 'BOM', 1030, 115),
    ('BOM', 'MAA', 1030, 115),
    ('IXM', 'BOM', 1180, 125),
    ('BOM', 'IXM', 1180, 125),
    ('TRZ', 'BOM', 1150, 120),
    ('BOM', 'TRZ', 1150, 120),

    # Tamil Nadu to Delhi
    ('CJB', 'DEL', 1950, 175),
    ('DEL', 'CJB', 1950, 175),
    ('MAA', 'DEL', 1760, 160),
    ('DEL', 'MAA', 1760, 160),
    ('IXM', 'DEL', 2050, 185),
    ('DEL', 'IXM', 2050, 185),
    ('TRZ', 'DEL', 1990, 180),
    ('DEL', 'TRZ', 1990, 180),

    # Tamil Nadu to Hyderabad
    ('CJB', 'HYD', 730, 85),
    ('HYD', 'CJB', 730, 85),
    ('MAA', 'HYD', 520, 70),
    ('HYD', 'MAA', 520, 70),
    ('IXM', 'HYD', 810, 95),
    ('HYD', 'IXM', 810, 95),

    # Tamil Nadu to Kochi, Goa, Pune, Kolkata
    ('CJB', 'GOI', 620, 75),
    ('GOI', 'CJB', 620, 75),
    ('CJB', 'PNQ', 910, 100),
    ('PNQ', 'CJB', 910, 100),
    ('MAA', 'COK', 560, 75),
    ('COK', 'MAA', 560, 75),
    ('MAA', 'GOI', 740, 85),
    ('GOI', 'MAA', 740, 85),
    ('MAA', 'PNQ', 910, 95),
    ('PNQ', 'MAA', 910, 95),
    ('MAA', 'CCU', 1370, 135),
    ('CCU', 'MAA', 1370, 135)
]

def generate_flight_dataset(num_records=28000):
    rows = []
    carrier_keys = list(CARRIERS.keys())
    carrier_probs = [CARRIERS[k]['market_share'] for k in carrier_keys]
    carrier_probs = np.array(carrier_probs) / sum(carrier_probs)
    
    route_weights = []
    for orig, dest, dist, time_mins in ROUTES:
        w = 1.0
        if 'CJB' in (orig, dest):
            w += 2.0
        if 'MAA' in (orig, dest):
            w += 2.5
        if 'IXM' in (orig, dest) or 'TRZ' in (orig, dest):
            w += 1.2
        route_weights.append(w)
    
    route_weights = np.array(route_weights) / sum(route_weights)
    weather_types = ['Clear', 'Partly Cloudy', 'Rain', 'Thunderstorm', 'Fog/Mist']

    for i in range(num_records):
        year = random.choice([2024, 2025])
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        day_of_week = (day + month * 2) % 7 + 1
        
        route_idx = np.random.choice(len(ROUTES), p=route_weights)
        origin, dest, distance, base_air_time = ROUTES[route_idx]
        
        if origin in ['SXV', 'TCR'] or dest in ['SXV', 'TCR']:
            carrier = np.random.choice(['9I', '6E', 'SG'], p=[0.6, 0.3, 0.1])
        else:
            carrier = np.random.choice(carrier_keys, p=carrier_probs)

        flight_num = f"{carrier}-{random.randint(100, 999)}"

        hour_weights = [
            0.01, 0.01, 0.01, 0.01, 0.02, 0.06,
            0.09, 0.10, 0.08, 0.07, 0.05, 0.04,
            0.04, 0.04, 0.04, 0.05, 0.06, 0.08,
            0.09, 0.08, 0.06, 0.04, 0.03, 0.01
        ]
        hour_weights = np.array(hour_weights) / sum(hour_weights)
        dep_hour = np.random.choice(range(24), p=hour_weights)
        dep_min = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        sched_dep_time = dep_hour * 100 + dep_min

        if 5 <= dep_hour < 9:
            time_slot = 'Morning Peak'
            slot_risk = 0.12
        elif 9 <= dep_hour < 16:
            time_slot = 'Midday'
            slot_risk = 0.08
        elif 16 <= dep_hour < 21:
            time_slot = 'Evening Peak'
            slot_risk = 0.22
        elif 21 <= dep_hour < 24:
            time_slot = 'Night'
            slot_risk = 0.15
        else:
            time_slot = 'Early Morning'
            slot_risk = 0.05

        is_tn_origin = origin in ['CJB', 'MAA', 'IXM', 'TRZ', 'TCR', 'SXV']
        is_tn_dest = dest in ['CJB', 'MAA', 'IXM', 'TRZ', 'TCR', 'SXV']
        is_delhi_route = 'DEL' in (origin, dest)

        if month in [10, 11, 12] and (is_tn_origin or is_tn_dest):
            w_probs = [0.35, 0.25, 0.25, 0.12, 0.03]
        elif month in [6, 7, 8, 9] and ('BOM' in (origin, dest) or 'COK' in (origin, dest) or 'CJB' in (origin, dest)):
            w_probs = [0.30, 0.30, 0.25, 0.10, 0.05]
        elif month in [12, 1] and is_delhi_route and dep_hour < 11:
            w_probs = [0.30, 0.20, 0.05, 0.02, 0.43]
        else:
            w_probs = [0.65, 0.20, 0.08, 0.04, 0.03]

        weather = np.random.choice(weather_types, p=w_probs)
        
        if weather == 'Clear':
            wind_speed = round(random.uniform(5, 15), 1)
            visibility_km = round(random.uniform(8, 10), 1)
            weather_delay_risk = 0.02
        elif weather == 'Partly Cloudy':
            wind_speed = round(random.uniform(8, 18), 1)
            visibility_km = round(random.uniform(6, 9), 1)
            weather_delay_risk = 0.05
        elif weather == 'Rain':
            wind_speed = round(random.uniform(15, 32), 1)
            visibility_km = round(random.uniform(3, 6), 1)
            weather_delay_risk = 0.30
        elif weather == 'Thunderstorm':
            wind_speed = round(random.uniform(25, 55), 1)
            visibility_km = round(random.uniform(1, 3.5), 1)
            weather_delay_risk = 0.65
        else:
            wind_speed = round(random.uniform(2, 10), 1)
            visibility_km = round(random.uniform(0.3, 1.8), 1)
            weather_delay_risk = 0.55

        carrier_unpunctuality = 1.0 - CARRIERS[carrier]['base_punctuality']
        origin_congestion = 0.15 if origin in ['DEL', 'BOM', 'BLR', 'MAA'] else 0.06
        dest_congestion = 0.14 if dest in ['DEL', 'BOM', 'BLR', 'MAA'] else 0.05
        rush_day_factor = 0.06 if day_of_week in [5, 7] else 0.0

        total_delay_prob = (
            carrier_unpunctuality * 0.40 +
            slot_risk * 0.30 +
            weather_delay_risk * 0.45 +
            (origin_congestion + dest_congestion) * 0.20 +
            rush_day_factor
        )
        total_delay_prob = min(max(total_delay_prob, 0.04), 0.96)

        is_delayed = 1 if np.random.rand() < total_delay_prob else 0

        if is_delayed == 1:
            if weather in ['Thunderstorm', 'Fog/Mist']:
                arr_delay = int(np.random.gamma(shape=4.0, scale=12.0) + 16)
            elif weather == 'Rain':
                arr_delay = int(np.random.gamma(shape=3.0, scale=8.0) + 16)
            elif slot_risk >= 0.20:
                arr_delay = int(np.random.gamma(shape=2.5, scale=7.0) + 16)
            else:
                arr_delay = int(np.random.gamma(shape=2.0, scale=6.0) + 16)
            arr_delay = min(arr_delay, 180)

            if weather in ['Thunderstorm', 'Fog/Mist', 'Rain'] and weather_delay_risk > 0.25:
                delay_reason = 'Weather'
            elif (origin_congestion + dest_congestion) > 0.25 and dep_hour in [8, 9, 18, 19, 20]:
                delay_reason = 'ATC Congestion'
            elif time_slot in ['Evening Peak', 'Night']:
                delay_reason = 'Late Inbound Aircraft'
            else:
                delay_reason = random.choice(['Airline Operations', 'Late Inbound Aircraft'])
        else:
            arr_delay = int(random.gauss(0, 6))
            arr_delay = max(min(arr_delay, 15), -20)
            delay_reason = 'On-Time'

        sched_duration = base_air_time + 15
        rows.append({
            'flight_id': f"FL-{20240000 + i}",
            'flight_number': flight_num,
            'carrier': carrier,
            'carrier_name': CARRIERS[carrier]['name'],
            'origin': origin,
            'origin_city': AIRPORTS[origin]['city'],
            'dest': dest,
            'dest_city': AIRPORTS[dest]['city'],
            'distance_km': distance,
            'year': year,
            'month': month,
            'day': day,
            'day_of_week': day_of_week,
            'sched_dep_time': sched_dep_time,
            'dep_hour': dep_hour,
            'dep_minute': dep_min,
            'time_slot': time_slot,
            'weather': weather,
            'wind_speed_kmh': wind_speed,
            'visibility_km': visibility_km,
            'sched_duration_min': sched_duration,
            'arr_delay_min': arr_delay,
            'delayed': is_delayed,
            'delay_reason': delay_reason
        })

    return pd.DataFrame(rows)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating comprehensive Indian & Tamil Nadu Flight Dataset...")
    df = generate_flight_dataset(28000)
    
    csv_path = os.path.join(data_dir, 'indian_flights_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} records to {csv_path}")

    processed_df = df[['year', 'month', 'day', 'day_of_week', 'dep_hour', 'carrier', 'origin', 'dest', 'distance_km', 'weather', 'wind_speed_kmh', 'visibility_km', 'delayed', 'arr_delay_min']]
    processed_path = os.path.join(data_dir, 'Processed_data15.csv')
    processed_df.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path}")

    airports_df = pd.DataFrame.from_dict(AIRPORTS, orient='index').reset_index().rename(columns={'index': 'iata_code'})
    airports_path = os.path.join(data_dir, 'airports_india.csv')
    airports_df.to_csv(airports_path, index=False)
    print(f"Saved Indian airports to {airports_path}")

    carriers_df = pd.DataFrame.from_dict(CARRIERS, orient='index').reset_index().rename(columns={'index': 'iata_code'})
    carriers_path = os.path.join(data_dir, 'carriers_india.csv')
    carriers_df.to_csv(carriers_path, index=False)
    print(f"Saved Indian carriers to {carriers_path}")

