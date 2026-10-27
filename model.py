"""
Model Training Script for Indian Flight Delay Prediction
Trains Random Forest Classifier and Delay Duration Regressor,
and saves model.pkl, model_rf.pkl, model_xgb.pkl, model_meta.pkl.
"""

from train_models import train_and_evaluate

if __name__ == '__main__':
    train_and_evaluate()
