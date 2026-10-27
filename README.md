# ✈️ AeroPredict: Indian Domestic Flight Delay Prediction & Analytics System
### Focused Case Study: Tamil Nadu Aviation Network (Coimbatore, Chennai, Madurai, Trichy)

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost-orange.svg)
![Standard](https://img.shields.io/badge/Standard-DGCA%2015--Min%20Threshold-emerald.svg)
![Region](https://img.shields.io/badge/Focus-Tamil%20Nadu%20%26%20India-indigo.svg)

---

## 📋 Executive Overview

**AeroPredict** is a complete, end-to-end Machine Learning web application and data analytics suite designed to forecast commercial flight delays across domestic Indian aviation corridors.

The project is customized and calibrated specifically for **Tamil Nadu airports** and their primary domestic connecting routes:
- **Coimbatore International Airport (CJB)** - Peelamedu (Western TN industrial & textile hub)
- **Chennai International Airport (MAA)** - Meenambakkam (Primary international and southern gateway)
- **Madurai Airport (IXM)** - South Tamil Nadu cultural & commercial hub
- **Tiruchirappalli International Airport (TRZ)** - Central Tamil Nadu gateway
- **Tuticorin (TCR)** & **Salem (SXV)** - Regional UDAN feeder connectivity
- **Connecting Metros**: Bengaluru (**BLR**), Mumbai (**BOM**), New Delhi (**DEL**), Hyderabad (**HYD**), Kochi (**COK**), Goa (**GOI**), Pune (**PNQ**), Kolkata (**CCU**)

All predictions adhere to the official **Directorate General of Civil Aviation (DGCA, India)** and **FAA international standard**: a flight is defined as **delayed** only if its arrival time exceeds scheduled arrival by **more than 15 minutes**.

---

## 🌟 Key System Highlights for Project Guides & Evaluators

1. **Authentic Indian Context**:
   - Replaced obsolete 2013 US data (Newark, JFK, LGA) with 28,000+ realistic domestic flight records.
   - Genuine Indian airline carriers: **IndiGo (6E)**, **Air India (AI)**, **Vistara (UK)**, **Akasa Air (QP)**, **SpiceJet (SG)**, and **Alliance Air (9I)**.

2. **Regional Meteorological Factors**:
   - Accounts for the **North-East Monsoon (Oct-Dec)** heavy coastal cloudbursts and convective storms impacting Chennai and coastal Tamil Nadu.
   - Models **Winter Morning Fog (Dec-Jan)** at Delhi (DEL) which causes cascading reactionary delays on aircraft arriving into Coimbatore and Chennai.

3. **Multi-Model Machine Learning Benchmark**:
   - **Random Forest Classifier**: Primary deployed model (Ensemble bagging with class balancing).
   - **XGBoost Classifier**: Gradient boosted trees with scale_pos_weight.
   - **Decision Tree**: Interpretable single-tree baseline.
   - **Logistic Regression**: Standard linear baseline.
   - **Delay Duration Regressor**: Continuous prediction of delay minutes (MAE: ~10.5 mins).

4. **Class Imbalance Resolution**:
   - In commercial aviation, ~77% of flights are on-time and ~23% are delayed.
   - Handled via `class_weight='balanced_subsample'`, `scale_pos_weight`, and calibrated decision thresholding at 42%.

5. **Explainable AI (XAI)**:
   - For every flight prediction, the system breaks down contributory factors (e.g. Evening peak turnaround congestion: +18%, Monsoon precipitation: +15%, Airline fleet punctuality: -10%).

---

## 🖥️ Multi-Page Dashboard Architecture

| Route | Page Name | Description |
|---|---|---|
| `/` | **Overview Dashboard** | Executive KPI stat cards, Monthly Monsoon trends (Chart.js), Airline reliability comparison, DGCA delay root causes donut chart, and 24-hr hourly risk heatmap. |
| `/predict` | **Delay Predictor** | Interactive prediction form with one-click test presets, circular probability gauge, delay duration range, and explainable AI factor breakdown. |
| `/routes` | **Routes & Airports Explorer** | Great-circle distance & flight time calculator, and airport infrastructure profiles for Tamil Nadu hubs and connecting metros. |
| `/guide` | **Model Benchmark & Viva Voce** | Detailed comparison table across all 4 ML algorithms, confusion matrix notes, feature importance ranking, and a structured Q&A viva cheat sheet for college guides. |
| `/api/predict` | **REST API (POST)** | JSON prediction endpoint for external integration. |
| `/api/stats` | **REST API (GET)** | JSON endpoint returning aggregate statistics and route performance. |
| `/api/airports`| **REST API (GET)** | JSON endpoint returning airport profiles and IATA metadata. |

---

## 📊 Machine Learning Model Benchmarks

| Algorithm / Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|---|---|---|---|---|---|---|
| **Random Forest (Balanced)** | **69.93%** | **32.81%** | **28.85%** | **30.70%** | **58.95%** | 🏆 **Primary Deployed** |
| **XGBoost (scale_pos)** | **58.73%** | **30.09%** | **59.47%** | **39.96%** | **61.86%** | ⚡ **High Recall** |
| **Decision Tree** | 54.96% | 27.53% | 58.24% | 37.39% | 58.18% | Baseline Tree |
| **Logistic Regression** | 56.09% | 29.04% | 62.49% | 39.66% | 62.23% | Linear Baseline |

*Delay Duration Regressor (Random Forest Regressor) Mean Absolute Error (MAE): **10.56 minutes**.*

### Feature Importance Hierarchy
1. **Atmospheric Wind Speed (km/h)**: ~16.9%
2. **Surface Visibility (km)**: ~16.3%
3. **Scheduled Departure Hour (Peak Slots)**: ~10.4%
4. **Flight Route Distance (km)**: ~9.1%
5. **Month (Monsoon Seasonality)**: ~8.8%
6. **Origin & Destination Airports**: ~13.6%
7. **Airline Carrier Punctuality**: ~4.2%

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment
Ensure Python 3.10+ is installed:
```powershell
# Clone or navigate to the repository
cd c:\Users\krish\Downloads\Flight-Delay-Prediction-master

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Application
Run from the project root:
```powershell
python run_app.py
```
Or run directly from the `Flight-Delay-Prediction` folder:
```powershell
python Flight-Delay-Prediction/app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Run Automated Endpoint Verification Tests
```powershell
python test_app_endpoints.py
```

### 4. Display Model Benchmark in Terminal
```powershell
python Flight-Delay-Prediction/test_accuracy.py
```

---

## 🎓 Guide Viva-Voce Cheat Sheet

* **Q: Why was the dataset customized to Tamil Nadu?**  
  *A: To make the project completely relatable, realistic, and defensible to Indian university guides and evaluators using local airports like Coimbatore (CJB) and Chennai (MAA).*

* **Q: Why is 15 minutes the cutoff for delay?**  
  *A: DGCA India and international aviation regulators (FAA/ICAO) define on-time performance within 15 minutes of scheduled gate arrival.*

* **Q: How does Delhi winter fog delay a Coimbatore flight?**  
  *A: Through cascading turnaround rotation delays (Reactionary Delay). Aircraft rotate across multiple sectors (e.g. DEL -> BOM -> CJB). A morning fog delay in Delhi ripples through the entire schedule for the rest of the day.*

---

## 🛠️ Technology Stack
- **Backend**: Python 3.13, Flask 3.1
- **Machine Learning**: Scikit-Learn, XGBoost, NumPy, Pandas
- **Frontend**: HTML5, Tailwind CSS, FontAwesome 6, Chart.js
- **Model Serialization**: Pickle (dual-head classification & regression)
