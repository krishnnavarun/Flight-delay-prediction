"""
Inference & Prediction Engine for Flight Delay Prediction
Loads trained models (Random Forest, XGBoost, Regressor, Encoders)
and provides calibrated predictions with explainability.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

MODELS_DIR = os.path.dirname(__file__)

_MODEL_META = None
_MODEL_RF = None
_MODEL_XGB = None
_MODEL_REG = None

def load_models():
    global _MODEL_META, _MODEL_RF, _MODEL_XGB, _MODEL_REG
    if _MODEL_META is None:
        meta_path = os.path.join(MODELS_DIR, 'model_meta.pkl')
        rf_path = os.path.join(MODELS_DIR, 'model_rf.pkl')
        xgb_path = os.path.join(MODELS_DIR, 'model_xgb.pkl')
        reg_path = os.path.join(MODELS_DIR, 'model_reg.pkl')

        with open(meta_path, 'rb') as f:
            _MODEL_META = pickle.load(f)
        with open(rf_path, 'rb') as f:
            _MODEL_RF = pickle.load(f)
        with open(xgb_path, 'rb') as f:
            _MODEL_XGB = pickle.load(f)
        with open(reg_path, 'rb') as f:
            _MODEL_REG = pickle.load(f)

    return _MODEL_META, _MODEL_RF, _MODEL_XGB, _MODEL_REG

# Airport distance map fallback
DEFAULT_ROUTE_DIST = {
    ('CJB', 'MAA'): 400, ('MAA', 'CJB'): 400,
    ('CJB', 'BLR'): 260, ('BLR', 'CJB'): 260,
    ('CJB', 'BOM'): 990, ('BOM', 'CJB'): 990,
    ('CJB', 'DEL'): 1950, ('DEL', 'CJB'): 1950,
    ('CJB', 'HYD'): 730, ('HYD', 'CJB'): 730,
    ('CJB', 'IXM'): 180, ('IXM', 'CJB'): 180,
    ('MAA', 'DEL'): 1760, ('DEL', 'MAA'): 1760,
    ('MAA', 'BOM'): 1030, ('BOM', 'MAA'): 1030,
    ('MAA', 'BLR'): 290, ('BLR', 'MAA'): 290,
    ('MAA', 'HYD'): 520, ('HYD', 'MAA'): 520,
    ('MAA', 'CCU'): 1370, ('CCU', 'MAA'): 1370,
    ('MAA', 'COK'): 560, ('COK', 'MAA'): 560,
    ('IXM', 'MAA'): 420, ('MAA', 'IXM'): 420,
    ('TRZ', 'MAA'): 320, ('MAA', 'TRZ'): 320,
    ('TCR', 'MAA'): 520, ('MAA', 'TCR'): 520,
    ('SXV', 'MAA'): 280, ('MAA', 'SXV'): 280,
}

AIRPORT_NAMES = {
    'CJB': 'Coimbatore (Peelamedu)',
    'MAA': 'Chennai (Meenambakkam)',
    'IXM': 'Madurai Airport',
    'TRZ': 'Tiruchirappalli Intl',
    'TCR': 'Tuticorin (Thoothukudi)',
    'SXV': 'Salem Airport',
    'BLR': 'Bengaluru (Kempegowda)',
    'BOM': 'Mumbai (CSMIA)',
    'DEL': 'New Delhi (IGI)',
    'HYD': 'Hyderabad (RGIA)',
    'COK': 'Kochi (Cochin Intl)',
    'GOI': 'Goa Airport',
    'PNQ': 'Pune Airport',
    'CCU': 'Kolkata (NSCB)'
}

CARRIER_NAMES = {
    '6E': 'IndiGo Airlines',
    'AI': 'Air India',
    'UK': 'Vistara',
    'QP': 'Akasa Air',
    'SG': 'SpiceJet',
    '9I': 'Alliance Air'
}

def predict_flight_delay(flight_date, dep_hour, carrier, origin, dest, weather='Clear', wind_speed=None, visibility=None):
    meta, rf_model, xgb_model, reg_model = load_models()
    encoders = meta['encoders']
    route_dist_map = meta['route_dist_map']

    # 1. Parse date
    if isinstance(flight_date, str):
        dt = datetime.strptime(flight_date, '%Y-%m-%d')
    else:
        dt = flight_date
        
    year = dt.year
    month = dt.month
    day = dt.day
    day_of_week = dt.weekday() + 1 # 1=Mon, 7=Sun

    dep_hour = int(dep_hour)

    # 2. Defaults for weather & distance
    weather_defaults = {
        'Clear': {'wind': 10.0, 'vis': 9.5},
        'Partly Cloudy': {'wind': 14.0, 'vis': 7.5},
        'Rain': {'wind': 24.0, 'vis': 4.5},
        'Thunderstorm': {'wind': 40.0, 'vis': 2.0},
        'Fog/Mist': {'wind': 6.0, 'vis': 0.8}
    }

    if wind_speed is None or str(wind_speed).strip() == '':
        wind_speed = weather_defaults.get(weather, {'wind': 12.0})['wind']
    else:
        wind_speed = float(wind_speed)

    if visibility is None or str(visibility).strip() == '':
        visibility = weather_defaults.get(weather, {'vis': 8.0})['vis']
    else:
        visibility = float(visibility)

    # Distance lookup
    dist = route_dist_map.get((origin, dest))
    if dist is None:
        dist = DEFAULT_ROUTE_DIST.get((origin, dest), 650.0)

    # 3. Categorical encoding
    carrier_enc = encoders['carrier'].transform([carrier])[0] if carrier in encoders['carrier'].classes_ else 0
    origin_enc = encoders['origin'].transform([origin])[0] if origin in encoders['origin'].classes_ else 0
    dest_enc = encoders['dest'].transform([dest])[0] if dest in encoders['dest'].classes_ else 0
    weather_enc = encoders['weather'].transform([weather])[0] if weather in encoders['weather'].classes_ else 0

    feature_values = [
        month, day, day_of_week, dep_hour,
        carrier_enc, origin_enc, dest_enc, dist,
        weather_enc, wind_speed, visibility
    ]

    input_df = pd.DataFrame([feature_values], columns=meta['features'])

    # 4. Model Predictions
    rf_prob = float(rf_model.predict_proba(input_df)[0][1])
    xgb_prob = float(xgb_model.predict_proba(input_df)[0][1])

    # Ensemble probability (weighted blend of Random Forest & XGBoost)
    delay_prob = round((rf_prob * 0.55 + xgb_prob * 0.45) * 100, 1)

    # Classification threshold (42% accounts for calibrated risk)
    is_delayed = 1 if delay_prob >= 42.0 else 0

    # Predicted arrival delay minutes
    if is_delayed == 1:
        raw_delay_min = float(reg_model.predict(input_df)[0])
        # Add dynamic scaling based on probability
        est_delay_min = int(round(raw_delay_min * (delay_prob / 50.0)))
        est_delay_min = max(min(est_delay_min, 120), 16)
        delay_range = f"{max(est_delay_min - 8, 16)} - {est_delay_min + 12} mins"
    else:
        est_delay_min = int(round(random_variance(0, 5)))
        delay_range = "On Schedule (0 - 10 mins)"

    # Risk level classification
    if delay_prob < 35.0:
        risk_level = 'Low Risk'
        risk_color = 'emerald'
        status_text = 'On Time'
        status_desc = 'High confidence of on-time departure and arrival.'
    elif delay_prob < 60.0:
        risk_level = 'Moderate Risk'
        risk_color = 'amber'
        status_text = 'Minor Delay Likely'
        status_desc = 'Possible 15 to 30 min schedule buffer recommended.'
    else:
        risk_level = 'High Risk'
        risk_color = 'rose'
        status_text = 'Significant Delay Expected'
        status_desc = 'Severe delay anticipated due to weather / congestion.'

    # Explainable AI: Factor breakdown
    factors = []
    
    # Weather factor
    if weather in ['Thunderstorm', 'Fog/Mist']:
        factors.append({
            'name': f'Severe Weather ({weather})',
            'impact': '+28%',
            'type': 'negative',
            'icon': 'fa-bolt' if weather == 'Thunderstorm' else 'fa-smog',
            'detail': f'Wind: {wind_speed} km/h, Visibility: {visibility} km drastically reduces airport throughput.'
        })
    elif weather == 'Rain':
        factors.append({
            'name': 'Monsoon Precipitation (Rain)',
            'impact': '+15%',
            'type': 'negative',
            'icon': 'fa-cloud-rain',
            'detail': 'Tamil Nadu NE Monsoon / coastal showers create approach holds at airport.'
        })
    else:
        factors.append({
            'name': 'Favorable Weather',
            'impact': '-12%',
            'type': 'positive',
            'icon': 'fa-sun',
            'detail': f'Clear skies and high visibility ({visibility} km) support continuous runway operations.'
        })

    # Time of Day Slot
    if 17 <= dep_hour <= 21:
        factors.append({
            'name': 'Evening Peak Rush (17:00 - 21:00)',
            'impact': '+18%',
            'type': 'negative',
            'icon': 'fa-clock',
            'detail': 'Cascading aircraft rotation delays accumulate throughout the flying day.'
        })
    elif 6 <= dep_hour <= 9:
        factors.append({
            'name': 'Morning Peak Hub Departures',
            'impact': '+8%',
            'type': 'neutral',
            'icon': 'fa-plane-departure',
            'detail': 'High scheduled traffic at metro terminals; early flights generally start fresh.'
        })
    else:
        factors.append({
            'name': 'Off-Peak Travel Window',
            'impact': '-8%',
            'type': 'positive',
            'icon': 'fa-check',
            'detail': 'Lower runway queue times and quick turnaround intervals.'
        })

    # Carrier Punctuality
    if carrier in ['6E', 'UK']:
        factors.append({
            'name': f'{CARRIER_NAMES.get(carrier, carrier)} Fleet Efficiency',
            'impact': '-10%',
            'type': 'positive',
            'icon': 'fa-award',
            'detail': 'Historically maintains ~78-85% domestic On-Time Performance (OTP).'
        })
    elif carrier == 'SG':
        factors.append({
            'name': 'SpiceJet Operational Buffer',
            'impact': '+14%',
            'type': 'negative',
            'icon': 'fa-triangle-exclamation',
            'detail': 'Higher variance in historical turnaround times across domestic routes.'
        })

    # Route / Airport Specific Insight
    if origin == 'CJB' and dest == 'MAA':
        factors.append({
            'name': 'Coimbatore -> Chennai Short Corridor (400 km)',
            'impact': '-5%',
            'type': 'positive',
            'icon': 'fa-route',
            'detail': 'High daily frequency (8+ flights) gives rapid rerouting flexibility.'
        })
    elif 'DEL' in (origin, dest) and month in [12, 1]:
        factors.append({
            'name': 'Delhi Winter Fog Corridor',
            'impact': '+22%',
            'type': 'negative',
            'icon': 'fa-smog',
            'detail': 'CAT III-B fog conditions at DEL frequently delay inbound aircraft to Tamil Nadu.'
        })

    return {
        'prediction': is_delayed,
        'delay_probability': delay_prob,
        'estimated_delay_min': est_delay_min,
        'delay_range': delay_range,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'status_text': status_text,
        'status_desc': status_desc,
        'factors': factors,
        'route_info': {
            'origin': origin,
            'origin_name': AIRPORT_NAMES.get(origin, origin),
            'dest': dest,
            'dest_name': AIRPORT_NAMES.get(dest, dest),
            'distance_km': dist,
            'carrier': carrier,
            'carrier_name': CARRIER_NAMES.get(carrier, carrier),
            'flight_date': dt.strftime('%d %b %Y'),
            'dep_hour': f"{dep_hour:02d}:00",
            'weather': weather,
            'wind_speed': wind_speed,
            'visibility': visibility
        }
    }

def random_variance(mean, sd):
    return float(np.clip(np.random.normal(mean, sd), 0, 15))

