"""
Analytics and Statistics Helper for Flight Delay Dashboard
Computes aggregate insights for the Tamil Nadu & Indian Domestic Flight Network.
"""

import os
import json
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), 'Data', 'indian_flights_data.csv')
AIRPORTS_PATH = os.path.join(os.path.dirname(__file__), 'Data', 'airports_india.csv')
CARRIERS_PATH = os.path.join(os.path.dirname(__file__), 'Data', 'carriers_india.csv')

_CACHED_DATA = None
_CACHED_STATS = None

def get_flight_dataframe():
    global _CACHED_DATA
    if _CACHED_DATA is None:
        if os.path.exists(DATA_PATH):
            _CACHED_DATA = pd.read_csv(DATA_PATH)
        else:
            # Fallback if accessed from root
            alt_path = os.path.join('Flight-Delay-Prediction', 'Data', 'indian_flights_data.csv')
            _CACHED_DATA = pd.read_csv(alt_path)
    return _CACHED_DATA

def get_dashboard_statistics():
    global _CACHED_STATS
    if _CACHED_STATS is not None:
        return _CACHED_STATS

    df = get_flight_dataframe()

    total_flights = len(df)
    delayed_flights = int(df['delayed'].sum())
    on_time_flights = total_flights - delayed_flights
    on_time_rate = round((on_time_flights / total_flights) * 100, 1)
    
    delayed_subset = df[df['delayed'] == 1]
    avg_delay_min = round(float(delayed_subset['arr_delay_min'].mean()), 1) if len(delayed_subset) > 0 else 24.5

    # 1. Monthly Performance & Monsoon Impact
    monthly_stats = df.groupby('month').agg(
        total=('flight_id', 'count'),
        delayed=('delayed', 'sum'),
        avg_delay=('arr_delay_min', 'mean')
    ).reset_index()
    monthly_stats['on_time_pct'] = ((1 - monthly_stats['delayed'] / monthly_stats['total']) * 100).round(1)
    monthly_stats['delay_pct'] = (monthly_stats['delayed'] / monthly_stats['total'] * 100).round(1)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct (NE Monsoon)', 'Nov (NE Monsoon)', 'Dec (Winter/Monsoon)']
    monthly_labels = month_names[:len(monthly_stats)]
    monthly_data = {
        'labels': monthly_labels,
        'on_time_pct': monthly_stats['on_time_pct'].tolist(),
        'delay_pct': monthly_stats['delay_pct'].tolist(),
        'avg_delay_min': monthly_stats['avg_delay'].round(1).tolist()
    }

    # 2. Airline Carrier Performance
    carrier_stats = df.groupby(['carrier', 'carrier_name']).agg(
        total=('flight_id', 'count'),
        delayed=('delayed', 'sum'),
        avg_delay=('arr_delay_min', 'mean')
    ).reset_index()
    carrier_stats['on_time_pct'] = ((1 - carrier_stats['delayed'] / carrier_stats['total']) * 100).round(1)
    carrier_stats = carrier_stats.sort_values(by='on_time_pct', ascending=False)
    
    best_carrier = carrier_stats.iloc[0]['carrier_name']

    carrier_data = {
        'labels': carrier_stats['carrier_name'].tolist(),
        'codes': carrier_stats['carrier'].tolist(),
        'on_time_pct': carrier_stats['on_time_pct'].tolist(),
        'delay_pct': (100 - carrier_stats['on_time_pct']).round(1).tolist(),
        'total_flights': carrier_stats['total'].tolist(),
        'avg_delay_min': carrier_stats['avg_delay'].round(1).tolist()
    }

    # 3. Delay Reasons Distribution (DGCA categories)
    reasons_df = df[df['delayed'] == 1]['delay_reason'].value_counts()
    reasons_data = {
        'labels': reasons_df.index.tolist(),
        'counts': reasons_df.values.tolist(),
        'percentages': (reasons_df / reasons_df.sum() * 100).round(1).tolist()
    }

    # 4. Hourly Departure Delay Risk (0 - 23 hours)
    hourly_stats = df.groupby('dep_hour').agg(
        total=('flight_id', 'count'),
        delayed=('delayed', 'sum')
    ).reset_index()
    hourly_stats['delay_rate_pct'] = (hourly_stats['delayed'] / hourly_stats['total'] * 100).round(1)
    
    hourly_data = {
        'hours': [f"{h:02d}:00" for h in hourly_stats['dep_hour']],
        'delay_rate_pct': hourly_stats['delay_rate_pct'].tolist(),
        'flight_volume': hourly_stats['total'].tolist()
    }

    # 5. Route Performance Highlights (Focus on Tamil Nadu)
    tn_airports = ['CJB', 'MAA', 'IXM', 'TRZ', 'TCR', 'SXV']
    tn_routes = df[df['origin'].isin(tn_airports) | df['dest'].isin(tn_airports)]
    route_stats = tn_routes.groupby(['origin', 'origin_city', 'dest', 'dest_city', 'distance_km']).agg(
        total_flights=('flight_id', 'count'),
        delayed_flights=('delayed', 'sum'),
        avg_delay=('arr_delay_min', 'mean')
    ).reset_index()
    route_stats['on_time_pct'] = ((1 - route_stats['delayed_flights'] / route_stats['total_flights']) * 100).round(1)
    route_stats['delay_pct'] = (100 - route_stats['on_time_pct']).round(1)
    route_stats = route_stats.sort_values(by='total_flights', ascending=False)

    top_routes = route_stats.head(15).to_dict(orient='records')
    worst_route = route_stats.sort_values(by='delay_pct', ascending=False).iloc[0]
    worst_route_name = f"{worst_route['origin']} ({worst_route['origin_city']}) to {worst_route['dest']} ({worst_route['dest_city']})"

    # Weather impact
    weather_stats = df.groupby('weather').agg(
        total=('flight_id', 'count'),
        delayed=('delayed', 'sum'),
        avg_delay=('arr_delay_min', 'mean')
    ).reset_index()
    weather_stats['delay_pct'] = (weather_stats['delayed'] / weather_stats['total'] * 100).round(1)

    _CACHED_STATS = {
        'summary': {
            'total_flights': f"{total_flights:,}",
            'delayed_flights': f"{delayed_flights:,}",
            'on_time_flights': f"{on_time_flights:,}",
            'on_time_rate': on_time_rate,
            'avg_delay_min': avg_delay_min,
            'best_carrier': best_carrier,
            'worst_route': worst_route_name
        },
        'monthly_data': monthly_data,
        'carrier_data': carrier_data,
        'reasons_data': reasons_data,
        'hourly_data': hourly_data,
        'top_routes': top_routes,
        'weather_impact': weather_stats.to_dict(orient='records')
    }
    return _CACHED_STATS

def get_airports_dict():
    df = get_flight_dataframe()
    origins = df[['origin', 'origin_city']].drop_duplicates().sort_values(by='origin')
    tn_codes = ['CJB', 'MAA', 'IXM', 'TRZ', 'TCR', 'SXV']
    
    airport_list = []
    # Add TN airports first
    for _, row in origins.iterrows():
        code = row['origin']
        city = row['origin_city']
        is_tn = code in tn_codes
        airport_list.append({
            'code': code,
            'city': city,
            'is_tn': is_tn,
            'display': f"{city} ({code}) {'[Tamil Nadu]' if is_tn else ''}"
        })
    # Sort TN first then alphabetical
    airport_list = sorted(airport_list, key=lambda x: (not x['is_tn'], x['city']))
    return airport_list

def get_carriers_dict():
    df = get_flight_dataframe()
    carriers = df[['carrier', 'carrier_name']].drop_duplicates().sort_values(by='carrier')
    return carriers.to_dict(orient='records')
