# ✈️ AeroPredict: PPT Presentation & Technical Project Defense Guide
### Indian Domestic Flight Delay Prediction System (Tamil Nadu Case Study)

---

## Slide 1: Title Slide
* **Project Title**: AeroPredict — Machine Learning-Powered Domestic Flight Delay Prediction & Analytics System
* **Domain**: Data Science, Machine Learning, Civil Aviation Analytics
* **Focus Corridor**: Tamil Nadu Aviation Network (Coimbatore `CJB`, Chennai `MAA`, Madurai `IXM`, Tiruchirappalli `TRZ`) & Connecting Metros (`BLR`, `BOM`, `DEL`, `HYD`)
* **Regulatory Standard**: Directorate General of Civil Aviation (DGCA, India) 15-Minute Threshold

---

## Slide 2: Problem Statement & Motivation
* **Industry Context**: Flight delays cause massive financial losses to airlines, severe passenger disruption, and cascading air traffic congestion.
* **Limitations of Existing Systems**: Many existing academic models rely on outdated US flight datasets (e.g., 2013 NYC flights), which fail to model Indian operational, infrastructural, and weather dynamics.
* **Our Solution**: An end-to-end predictive system tailored for Indian domestic aviation that forecasts delay probabilities, estimates delay durations in minutes, and explains root causal factors.

---

## Slide 3: Regional Scope & Domain Realism
* **Key Tamil Nadu Airports**:
  * Coimbatore International Airport (`CJB`) — Western TN industrial corridor
  * Chennai International Airport (`MAA`) — Primary international & South India gateway
  * Madurai Airport (`IXM`) — Cultural & southern trade hub
  * Tiruchirappalli International Airport (`TRZ`) — Central TN & Southeast Asia connector
  * Tuticorin (`TCR`) & Salem (`SXV`) — Regional UDAN connectivity
* **Domestic Airlines Tracked**: IndiGo (`6E`), Air India (`AI`), Vistara (`UK`), Akasa Air (`QP`), SpiceJet (`SG`), Alliance Air (`9I`).
* **Indian Weather Factors**:
  * **North-East Monsoon (Oct–Dec)**: Heavy coastal rain & cloudbursts impacting Chennai & Coimbatore.
  * **Delhi Winter Fog (Dec–Jan)**: CAT III-B visibility drops at DEL causing cascading delays across southern domestic sectors.

---

## Slide 4: Regulatory Standard — DGCA 15-Minute Rule
* In civil aviation, flights rarely land at the exact second of schedule due to taxiing, runway sequencing, and air holds.
* **Official DGCA & FAA/ICAO Benchmark**:
  * Delay $\le$ 15 minutes $\rightarrow$ **Class 0: On-Time**
  * Delay $>$ 15 minutes $\rightarrow$ **Class 1: Delayed**
* Prevents false alarms on minor 5–10 minute taxi variance.

---

## Slide 5: Dataset & Preprocessing Pipeline
* **Dataset Scale**: 28,000 authentic domestic flight records spanning 12 operational months.
* **Input Features (11 Attributes)**:
  1. `month` (Monsoon Seasonality: Jun–Sep SW Monsoon, Oct–Dec NE Monsoon)
  2. `day` & `day_of_week` (Travel surges, weekend traffic)
  3. `dep_hour` (Peak rush: 06:00–09:00 morning & 17:00–21:00 evening)
  4. `carrier` (Airline fleet reliability)
  5. `origin` & `dest` (Airport capacity & runway holds)
  6. `distance_km` (Great-circle route flight distance)
  7. `weather` (Clear, Partly Cloudy, Rain, Thunderstorm, Fog/Mist)
  8. `wind_speed_kmh` (Atmospheric wind speed)
  9. `visibility_km` (Surface runway visibility)
* **Preprocessing**: LabelEncoding for categoricals, StandardScaler for linear models, 80:20 Stratified Train-Test split.

---

## Slide 6: Handling Class Imbalance
* **Real-World Imbalance**: In commercial aviation, ~77% of flights are on-time and ~23% are delayed (3.3:1 ratio).
* **Naive Model Pitfall**: An uncalibrated model predicting "On-Time" for every flight scores 77% accuracy while failing 100% on delay detection.
* **Our Methodological Solution**:
  1. Random Forest: `class_weight='balanced_subsample'`
  2. XGBoost: `scale_pos_weight = N_neg / N_pos`
  3. Calibrated decision thresholding at 42% for risk sensitivity.

---

## Slide 7: Machine Learning Algorithms Evaluated
1. **Random Forest Classifier (Primary Model)**:
   * Bagging ensemble of 120 decision trees.
   * Handles non-linear feature interactions (Weather $\times$ Evening Slot $\times$ Turnaround).
2. **XGBoost Classifier (High Recall Model)**:
   * Gradient boosted sequential decision trees minimizing log-loss with regularized learning rate ($\eta = 0.08$).
3. **Decision Tree Classifier**:
   * Interpretable single-tree white-box baseline.
4. **Logistic Regression**:
   * Standard linear classification baseline.
5. **Random Forest Regressor (Secondary Model)**:
   * Predicts continuous arrival delay duration in minutes.

---

## Slide 8: Model Performance Benchmark Results
*Evaluated on 5,600 test flights (20% Stratified Split):*

| Algorithm | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Role |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Random Forest (Balanced)** | **69.93%** | **32.81%** | **28.85%** | **30.70%** | **58.95%** | 🏆 **Primary Deployed** |
| **XGBoost (scale_pos)** | **58.73%** | **30.09%** | **59.47%** | **39.96%** | **61.86%** | ⚡ **High Recall on Delays** |
| **Decision Tree** | 54.96% | 27.53% | 58.24% | 37.39% | 58.18% | Tree Baseline |
| **Logistic Regression** | 56.09% | 29.04% | 62.49% | 39.66% | 62.23% | Linear Baseline |

* **Delay Duration Regressor**: Mean Absolute Error (MAE) = **10.56 minutes**.

---

## Slide 9: Feature Importance & Explainable AI (XAI)
* **Top Contributing Factors (Gini Impurity Decrease)**:
  1. Wind Speed (`16.88%`) & Visibility (`16.25%`) $\rightarrow$ Primary meteorological constraints.
  2. Day of Month (`11.31%`) & Departure Hour (`10.39%`) $\rightarrow$ Peak turnaround accumulation.
  3. Route Distance (`9.05%`) & Month (`8.75%`) $\rightarrow$ Monsoon exposure.
  4. Origin & Destination Congestion (`13.61%`).
  5. Carrier Reliability (`4.21%`).
* **XAI Decomposition**: For each prediction, the system displays positive and negative contributing factors (e.g., Evening Peak `+18%`, Monsoon Rain `+15%`, IndiGo Efficiency `-10%`).

---

## Slide 10: System Architecture & Web UI
* **Clean Corporate Light UI**: Built using HTML5, Tailwind CSS, FontAwesome 6, and Chart.js.
* **Core Application Pages**:
  1. `/` — Executive Dashboard (KPI stat cards, Monthly Monsoon trends, Carrier reliability, 24-hr risk curve, route table).
  2. `/predict` — Interactive Predictor (Travel date, slot, carrier, airports, weather inputs, real-time risk gauge).
  3. `/routes` — Corridors Explorer (Great-circle calculator, airport infrastructure profiles for CJB, MAA, IXM, TRZ).
  4. `/guide` — Evaluation benchmark & viva defense cheat sheet.
  5. `/api/predict` & `/api/stats` — Programmatic REST API endpoints.

---

## Slide 11: Technology Stack
* **Language & Core**: Python 3.13, NumPy, Pandas, SciPy
* **Machine Learning**: Scikit-Learn 1.9, XGBoost 3.4, Joblib
* **Web Framework**: Flask 3.1, Jinja2, Werkzeug
* **Data Visualization**: Chart.js, Matplotlib, Seaborn
* **Deployment**: Windows / Linux compatible, standalone batch launcher (`run.bat`)

---

## Slide 12: Guide Viva-Voce Defense (Top Q&A)
* **Q: Why Tamil Nadu routes?**  
  *A: Replaces obsolete US data with realistic domestic aviation, incorporating regional weather like the North-East Monsoon.*
* **Q: Why is 15 minutes chosen as the threshold?**  
  *A: Conforms to DGCA India and international FAA/ICAO standards for official delay classification.*
* **Q: Why does Delhi fog affect Coimbatore flights?**  
  *A: Reactionary delay. Aircraft fly multi-leg rotations (e.g. DEL $\rightarrow$ BOM $\rightarrow$ CJB). A morning fog delay in Delhi ripples through subsequent flights all day.*
* **Q: Why use Random Forest & XGBoost?**  
  *A: Flight delay causality is non-linear; tree ensembles model complex feature interactions (weather $\times$ peak hour $\times$ turnaround) without strict linear assumptions.*

---

## Slide 13: Conclusion & Future Scope
* **Achievements**: Successfully built an authentic, end-to-end domestic aviation delay prediction system with high explainability and clean enterprise UI.
* **Future Enhancements**:
  * Real-time live weather API integration (e.g., OpenWeatherMap / IMD radar).
  * Live ADS-B radar flight tracking feed integration.
  * Deep Learning models (LSTM / GRU) for time-series aircraft tail-number tracking.
