"""
Verification script using Flask Test Client to probe all routes and endpoints.
"""

import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from app import app

def test_endpoints():
    client = app.test_client()

    print("Testing AeroPredict Web Application Routes & Endpoints...\n")

    # 1. Dashboard Overview
    r1 = client.get('/')
    assert r1.status_code == 200, f"Dashboard failed: {r1.status_code}"
    assert b"Domestic Flight Delay" in r1.data, "Dashboard title missing"
    assert b"Coimbatore" in r1.data, "Coimbatore mention missing"
    print("[PASS] Route 1: '/' Dashboard Overview returned HTTP 200 OK")

    # 2. Predictor GET Page
    r2 = client.get('/predict')
    assert r2.status_code == 200, f"Predict page failed: {r2.status_code}"
    assert b"Flight Delay Predictor" in r2.data
    print("[PASS] Route 2: '/predict' (GET) Predictor Page returned HTTP 200 OK")

    # 3. Predictor POST (Form submission)
    r3 = client.post('/predict', data={
        'flight_date': '2025-11-20',
        'dep_hour': '18',
        'carrier': '6E',
        'origin': 'CJB',
        'dest': 'MAA',
        'weather': 'Rain',
        'wind_speed': '24.0',
        'visibility': '4.5'
    })
    assert r3.status_code == 200, f"Predict POST failed: {r3.status_code}"
    assert b"Delay Risk" in r3.data or b"Risk" in r3.data
    print("[PASS] Route 3: '/predict' (POST) Form Inference returned HTTP 200 OK with Delay Probability")

    # 4. Routes Explorer
    r4 = client.get('/routes')
    assert r4.status_code == 200, f"Routes page failed: {r4.status_code}"
    assert b"Tamil Nadu Airport Network Profiles" in r4.data
    print("[PASS] Route 4: '/routes' Routes Explorer returned HTTP 200 OK")

    # 5. Guide Viva Voce Hub
    r5 = client.get('/guide')
    assert r5.status_code == 200, f"Guide page failed: {r5.status_code}"
    assert b"Viva-Voce Cheat Sheet" in r5.data
    assert b"Random Forest" in r5.data
    print("[PASS] Route 5: '/guide' Model Benchmark & Viva Guide returned HTTP 200 OK")

    # 6. REST API: /api/predict
    r6 = client.post('/api/predict', json={
        'flight_date': '2025-12-25',
        'dep_hour': 8,
        'carrier': 'AI',
        'origin': 'CJB',
        'dest': 'DEL',
        'weather': 'Fog/Mist',
        'wind_speed_kmh': 6.0,
        'visibility_km': 0.8
    })
    assert r6.status_code == 200, f"API predict failed: {r6.status_code}"
    json_data = r6.get_json()
    assert json_data['success'] is True, "API success is False"
    prob = json_data['data']['delay_probability']
    status = json_data['data']['status_text']
    print(f"[PASS] Route 6: '/api/predict' (POST JSON) returned HTTP 200 OK (Status: {status}, Prob: {prob}%)")

    # 7. REST API: /api/stats
    r7 = client.get('/api/stats')
    assert r7.status_code == 200, f"API stats failed: {r7.status_code}"
    stats_json = r7.get_json()
    assert stats_json['success'] is True
    total_f = stats_json['data']['summary']['total_flights']
    print(f"[PASS] Route 7: '/api/stats' (GET JSON) returned HTTP 200 OK (Total Monitored Flights: {total_f})")

    # 8. REST API: /api/airports
    r8 = client.get('/api/airports')
    assert r8.status_code == 200
    airports_json = r8.get_json()
    assert airports_json['success'] is True
    print(f"[PASS] Route 8: '/api/airports' (GET JSON) returned HTTP 200 OK ({len(airports_json['data'])} Airports listed)")

    print("\nALL 8 ENDPOINTS TESTED AND VERIFIED SUCCESSFULLY!")

if __name__ == '__main__':
    test_endpoints()

