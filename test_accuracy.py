import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/Processed_data15.csv')

le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])
le_dest = LabelEncoder()
df['dest'] = le_dest.fit_transform(df['dest'])
le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

X = df.iloc[:, 0:6].values
y = df['delayed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18)

print("Training RandomForest (Current)...")
clf = RandomForestClassifier(random_state=18, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred))

print("\nTraining XGBoost (New)...")
from xgboost import XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=18, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

