"""
AeroPredict: Indian Domestic Flight Delay Prediction System
Focus Area: Tamil Nadu Hubs (Coimbatore, Chennai, Madurai, Trichy) and Connected Metros

Routes:
- /            : Executive Overview Dashboard with interactive charts & route metrics
- /predict     : Interactive Flight Delay Predictor with Dual-Ensemble ML & XAI
- /routes      : Route & Airport Explorer for Tamil Nadu Aviation Network
- /guide       : Technical Evaluation Benchmark & Guide Viva Voce Hub
- /api/predict : JSON REST API for flight delay prediction
- /api/stats   : JSON REST API for dashboard statistics
"""

import os
import sys
import pickle
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for

# Ensure current directory is on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from analytics_helper import get_dashboard_statistics, get_airports_dict, get_carriers_dict
from predictor_engine import predict_flight_delay, load_models

app = Flask(__name__, 
            template_folder=os.path.join(CURRENT_DIR, 'templates'),
            static_folder=os.path.join(CURRENT_DIR, 'static'))

# Preload models at startup
try:
    load_models()
    print("[INFO] AeroPredict Machine Learning Models successfully initialized.")
except Exception as e:
    print(f"[WARNING] Could not preload models: {e}")

@app.after_request
def add_cache_control_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ==============================================================================
# WEB PAGE ROUTES
# ==============================================================================

@app.route('/')
def dashboard():
    """Main Executive Overview Dashboard."""
    try:
        stats = get_dashboard_statistics()
        return render_template('dashboard.html', stats=stats, active_page='dashboard')
    except Exception as e:
        return f"Error loading dashboard: {e}", 500

@app.route('/predict', methods=['GET', 'POST'], endpoint='predict')
@app.route('/predict_page', methods=['GET', 'POST'], endpoint='predict_page')
def predict_page():
    """Flight Delay Predictor Page with dual-ensemble ML inference."""
    if request.method == 'POST':
        try:
            flight_date = request.form.get('flight_date')
            dep_hour = request.form.get('dep_hour', '8')
            carrier = request.form.get('carrier')
            origin = request.form.get('origin')
            dest = request.form.get('dest')
            weather = request.form.get('weather', 'Clear')
            wind_speed = request.form.get('wind_speed')
            visibility = request.form.get('visibility')

            if not flight_date or not carrier or not origin or not dest:
                return render_template('predict.html', 
                                       error="Please provide travel date, carrier, origin, and destination.",
                                       active_page='predict')

            if origin == dest:
                return render_template('predict.html', 
                                       error="Origin and Destination airports must be different.",
                                       active_page='predict')

            # Run ML prediction
            result = predict_flight_delay(
                flight_date=flight_date,
                dep_hour=dep_hour,
                carrier=carrier,
                origin=origin,
                dest=dest,
                weather=weather,
                wind_speed=wind_speed,
                visibility=visibility
            )

            form_data = {
                'flight_date': flight_date,
                'dep_hour': dep_hour,
                'carrier': carrier,
                'origin': origin,
                'dest': dest,
                'weather': weather,
                'wind_speed': wind_speed,
                'visibility': visibility
            }

            return render_template('predict.html', result=result, form_data=form_data, active_page='predict')

        except Exception as e:
            return render_template('predict.html', error=f"Prediction error: {str(e)}", active_page='predict')

    # GET request - default blank predictor
    return render_template('predict.html', active_page='predict')

# Backwards compatible alias for old /predict POST form submission
@app.route('/predict_legacy', methods=['POST'])
def predict_legacy():
    return redirect(url_for('predict_page'))

@app.route('/routes')
def routes_page():
    """Tamil Nadu & Indian Airport Corridors Explorer."""
    try:
        stats = get_dashboard_statistics()
        airports = get_airports_dict()
        return render_template('routes.html', stats=stats, airports=airports, active_page='routes')
    except Exception as e:
        return f"Error loading routes explorer: {e}", 500

@app.route('/guide')
def guide_page():
    """Model Technical Evaluation & College Project Guide Viva Voce Hub."""
    try:
        meta, _, _, _ = load_models()
        return render_template('guide.html', meta=meta, active_page='guide')
    except Exception as e:
        return f"Error loading viva notes: {e}", 500

# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for programmatic flight delay prediction.
    Example payload:
    {
      "flight_date": "2025-11-20",
      "dep_hour": 18,
      "carrier": "6E",
      "origin": "CJB",
      "dest": "MAA",
      "weather": "Rain",
      "wind_speed_kmh": 24.0,
      "visibility_km": 4.5
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'JSON payload required'}), 400

        flight_date = data.get('flight_date', datetime.today().strftime('%Y-%m-%d'))
        dep_hour = data.get('dep_hour', 8)
        carrier = data.get('carrier', '6E')
        origin = data.get('origin', 'CJB')
        dest = data.get('dest', 'MAA')
        weather = data.get('weather', 'Clear')
        wind_speed = data.get('wind_speed_kmh')
        visibility = data.get('visibility_km')

        if origin == dest:
            return jsonify({'success': False, 'error': 'Origin and Destination must be distinct.'}), 400

        result = predict_flight_delay(
            flight_date=flight_date,
            dep_hour=dep_hour,
            carrier=carrier,
            origin=origin,
            dest=dest,
            weather=weather,
            wind_speed=wind_speed,
            visibility=visibility
        )

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """JSON API endpoint returning dashboard statistics."""
    try:
        stats = get_dashboard_statistics()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/airports', methods=['GET'])
def api_airports():
    """JSON API endpoint returning airport directory."""
    try:
        airports = get_airports_dict()
        return jsonify({'success': True, 'data': airports})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting AeroPredict Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
