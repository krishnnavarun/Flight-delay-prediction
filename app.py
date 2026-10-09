import os
import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Precompute LabelEncoders once on startup to avoid doing it on every request
try:
    df = pd.read_csv('Data/Processed_data15.csv')
    le_carrier = LabelEncoder()
    le_carrier.fit(df['carrier'])
    le_dest = LabelEncoder()
    le_dest.fit(df['dest'])
    le_origin = LabelEncoder()
    le_origin.fit(df['origin'])
    
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or data: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        flight_date = request.form.get('flight_date')
        carrier = request.form.get('carrier')
        origin = request.form.get('origin')
        dest = request.form.get('dest')

        if not flight_date or not carrier or not origin or not dest:
            return render_template('index.html', error="Please fill all fields to predict the flight delay.")

        year, month, day = flight_date.split('-')
        year = int(year)
        month = int(month)
        day = int(day)
        
        # Prepare data for prediction
        input_data = pd.DataFrame([[year, month, day, carrier, origin, dest]], 
                                  columns=['year', 'month', 'date', 'carrier', 'origin', 'dest'])
        
        # Transform using fitted label encoders
        input_data['carrier'] = le_carrier.transform(input_data['carrier'])
        input_data['origin'] = le_origin.transform(input_data['origin'])
        input_data['dest'] = le_dest.transform(input_data['dest'])
        
        x = input_data.iloc[:, :6].values
        prediction = model.predict(x)[0] # Extract the single prediction value
        
        return render_template('index.html', prediction_text=prediction, 
                               flight_date=flight_date, carrier=carrier, 
                               origin=origin, dest=dest)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)