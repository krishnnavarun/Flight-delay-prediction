"""
Machine Learning Training Pipeline for Indian Domestic Flight Delay Prediction
Trains and evaluates:
1. Random Forest Classifier (with class balancing for realistic flight delay detection)
2. XGBoost Classifier (with scale_pos_weight)
3. Decision Tree Classifier
4. Logistic Regression
5. Random Forest Regressor (Delay Duration Predictor in minutes)

Saves models and comprehensive evaluation metrics for presentation to project guides.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'Data', 'indian_flights_data.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} flight records from {data_path}")

    # Feature engineering & Selection
    features = [
        'month', 'day', 'day_of_week', 'dep_hour',
        'carrier', 'origin', 'dest', 'distance_km',
        'weather', 'wind_speed_kmh', 'visibility_km'
    ]
    
    X = df[features].copy()
    y_class = df['delayed'].values
    y_reg = df['arr_delay_min'].values

    # Categorical encoders
    encoders = {}
    cat_cols = ['carrier', 'origin', 'dest', 'weather']
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Route distance lookup for inference
    route_dist_map = {}
    for _, row in df[['origin', 'dest', 'distance_km']].drop_duplicates().iterrows():
        route_dist_map[(row['origin'], row['dest'])] = float(row['distance_km'])

    # Train-test split (80-20 stratified)
    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )

    # Calculate class imbalance ratio
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = neg_count / pos_count
    print(f"Class distribution: On-Time={neg_count}, Delayed={pos_count} (Ratio={imbalance_ratio:.2f})")

    print("\n--- Training Models with Class Imbalance Compensation ---")
    
    # 1. Random Forest Classifier (Balanced)
    print("1. Training Random Forest Classifier (Balanced)...")
    rf_clf = RandomForestClassifier(
        n_estimators=120, max_depth=16, min_samples_split=4,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.40).astype(int)

    # 2. XGBoost Classifier (Balanced)
    print("2. Training XGBoost Classifier (scale_pos_weight)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=120, max_depth=6, learning_rate=0.08,
        scale_pos_weight=imbalance_ratio,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_prob_xgb >= 0.45).astype(int)

    # 3. Decision Tree
    print("3. Training Decision Tree Classifier...")
    dt_clf = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    dt_clf.fit(X_train, y_train)
    y_prob_dt = dt_clf.predict_proba(X_test)[:, 1]
    y_pred_dt = (y_prob_dt >= 0.45).astype(int)

    # 4. Logistic Regression
    print("4. Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_clf.fit(X_train_scaled, y_train)
    y_prob_lr = lr_clf.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.45).astype(int)

    # 5. Delay Regressor (Predicting minutes for delayed flights)
    print("5. Training Delay Duration Regressor (Random Forest Regressor)...")
    delayed_mask_train = y_train == 1
    delayed_mask_test = y_test == 1
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train[delayed_mask_train], y_reg_train[delayed_mask_train])
    reg_preds = rf_reg.predict(X_test[delayed_mask_test])
    mae = np.mean(np.abs(reg_preds - y_reg_test[delayed_mask_test]))
    print(f"Delay Regressor MAE on delayed flights: {mae:.2f} minutes")

    # Metrics Compilation
    models_metrics = {
        'Random Forest': {
            'accuracy': round(accuracy_score(y_test, y_pred_rf) * 100, 2),
            'precision': round(precision_score(y_test, y_pred_rf) * 100, 2),
            'recall': round(recall_score(y_test, y_pred_rf) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred_rf) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_prob_rf) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist()
        },
        'XGBoost': {
            'accuracy': round(accuracy_score(y_test, y_pred_xgb) * 100, 2),
            'precision': round(precision_score(y_test, y_pred_xgb) * 100, 2),
            'recall': round(recall_score(y_test, y_pred_xgb) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred_xgb) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_prob_xgb) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist()
        },
        'Decision Tree': {
            'accuracy': round(accuracy_score(y_test, y_pred_dt) * 100, 2),
            'precision': round(precision_score(y_test, y_pred_dt) * 100, 2),
            'recall': round(recall_score(y_test, y_pred_dt) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred_dt) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_prob_dt) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred_dt).tolist()
        },
        'Logistic Regression': {
            'accuracy': round(accuracy_score(y_test, y_pred_lr) * 100, 2),
            'precision': round(precision_score(y_test, y_pred_lr) * 100, 2),
            'recall': round(recall_score(y_test, y_pred_lr) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred_lr) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_prob_lr) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
        }
    }

    # Feature Importances from Random Forest
    rf_importances = dict(zip(features, [round(float(val) * 100, 2) for val in rf_clf.feature_importances_]))
    rf_importances = dict(sorted(rf_importances.items(), key=lambda item: item[1], reverse=True))

    metadata = {
        'features': features,
        'cat_cols': cat_cols,
        'encoders': encoders,
        'route_dist_map': route_dist_map,
        'models_metrics': models_metrics,
        'feature_importances': rf_importances,
        'regressor_mae': round(float(mae), 2)
    }

    with open(os.path.join(base_dir, 'model_meta.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        
    with open(os.path.join(base_dir, 'model_rf.pkl'), 'wb') as f:
        pickle.dump(rf_clf, f)

    with open(os.path.join(base_dir, 'model_xgb.pkl'), 'wb') as f:
        pickle.dump(xgb_clf, f)

    with open(os.path.join(base_dir, 'model_reg.pkl'), 'wb') as f:
        pickle.dump(rf_reg, f)

    with open(os.path.join(base_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(rf_clf, f)

    print("\nAll models and metadata successfully trained and serialized to disk!")

if __name__ == '__main__':
    train_and_evaluate()

