import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier
import joblib

df_train = pd.read_csv("data/processed/train.csv")
df_val = pd.read_csv("data/processed/val.csv")
df_test = pd.read_csv("data/processed/test.csv")

X_train = df_train.drop(columns=["claim_number", "fraud"])
y_train = df_train["fraud"]
X_val = df_val.drop(columns=["claim_number", "fraud"])
y_val = df_val["fraud"]
X_test = df_test.drop(columns=["claim_number"])

categorical_features = X_train.columns[X_train.dtypes == object].tolist()
numeric_features = X_train.columns[X_train.dtypes != object].tolist()

column_transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown='ignore'), categorical_features),
        ("minmax", MinMaxScaler(), numeric_features),
    ],
    remainder="passthrough",
)

param_grid = {
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    "gamma": [0, 0.25, 0.5, 1.0],
    "n_estimators": [10, 20, 40, 60, 80, 100, 150, 200],
}

xgb_clf = RandomizedSearchCV(
    XGBClassifier(),
    param_distributions=param_grid,
    n_iter=50,
    n_jobs=-1,
    cv=5,
    random_state=23,
    scoring="roc_auc",
)

pipeline = make_pipeline(column_transformer, SMOTE(), xgb_clf)
pipeline.fit(X_train, y_train)

y_val_pred = pipeline.predict_proba(X_val)[:, 1]
metric = roc_auc_score(y_val, y_val_pred)

if isinstance(pipeline[-1], RandomizedSearchCV) or isinstance(pipeline[-1], GridSearchCV):
    print(f"Best params: {pipeline[-1].best_params_}")

print(f"AUC score: {metric}")

best_model = pipeline[-1].best_estimator_

# Transforming test data
X_test_encoded = column_transformer.transform(X_test)

# Setting enable_categorical=True for XGBoost prediction
best_model.set_params(**{'enable_categorical': True})
y_test_pred = best_model.predict_proba(X_test_encoded)[:, 1]

df_submission = pd.DataFrame({
    "claim_number": df_test["claim_number"],
    "fraud": y_test_pred
})

df_submission.to_csv("data/submission.csv", index=False)
joblib.dump(best_model, 'xgb_model.pkl')

input_data = {
    "claim_number": [29609],
    "age_of_driver": [50.0],
    "gender": ["F"],
    "marital_status": [1.0],
    "safty_rating": [90],
    "annual_income": [39135.0],
    "high_education_ind": [1],
    "address_change_ind": [1],
    "living_status": ["Own"],
    "accident_site": ["Local"],
    "past_num_of_claims": [0],
    "witness_present_ind": [0.0],
    "liab_prct": [50],
    "channel": ["Online"],
    "policy_report_filed_ind": [0],
    "claim_est_payout": [5866.835619],
    "age_of_vehicle": [5.0],
    "vehicle_category": ["Large"],
    "vehicle_price": [9556.595872],
    "vehicle_weight": [28996.57016],
    "latitude": [40.66],
    "longitude": [-80.24]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

X_input_encoded = column_transformer.transform(input_df)

# Use the trained model to make predictions
best_model.set_params(**{'enable_categorical': True})
fraud_probability = best_model.predict_proba(X_input_encoded)[:, 1]

result_df = pd.DataFrame({
    "claim_number": input_df["claim_number"],
    "fraud_probability": fraud_probability
})

# Display the results
print(result_df)


