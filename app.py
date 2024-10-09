from flask import Flask, render_template, request, request, redirect, url_for, send_from_directory,flash
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
import os
import glob
from classify import prediction
import tensorflow as tf
import  _thread
import time
from werkzeug.utils import secure_filename



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predicted():

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
    # Collect user inputs from the form
    claim_number = int(request.form['claim'])
    age_of_driver = float(request.form['age_of_driver'])
    gender = request.form['gender']
    marital_status_str = request.form['marital_status']
    marital_status = 1 if marital_status_str.lower() == 'yes' else 0
    safty_rating = int(request.form['safty_rating'])
    annual_income = float(request.form['annual_income'])
    high_education_ind_str = request.form['high_education_ind']
    high_education_ind = 1 if high_education_ind_str.lower() == 'yes' else 0
    address_change_ind_str = request.form['address_change_ind']
    address_change_ind = 1 if address_change_ind_str.lower() == 'yes' else 0
    living_status = request.form['living_status']
    accident_site = request.form['accident_site']
    past_num_of_claims = int(request.form['past_num_of_claims'])
    witness_present_ind = float(request.form['witness_present_ind'])
    liab_prct = int(request.form['liab_prct'])
    channel = request.form['channel']
    policy_report_filed_ind_str = request.form['policy_report_filed_ind']
    policy_report_filed_ind = 1 if policy_report_filed_ind_str.lower() == 'yes' else 0
    claim_est_payout = float(request.form['claim_est_payout'])
    age_of_vehicle = float(request.form['age_of_vehicle'])
    vehicle_category = request.form['vehicle_category']
    vehicle_price = float(request.form['vehicle_price'])
    vehicle_weight = float(request.form['vehicle_weight'])
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])

    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        filename  = str(len(os.listdir(app.config['UPLOAD_FOLDER']))+1)+'.jpg'
        # Move the file form the temporal folder to
        # the upload folder we setup
        file_name_full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_name_full_path)
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        

    # Create a dictionary with the user input data
    input_data = {
        "claim_number": [claim_number],
        "age_of_driver": [age_of_driver],
        "gender": [gender],
        "marital_status": [marital_status],
        "safty_rating": [safty_rating],
        "annual_income": [annual_income],
        "high_education_ind": [high_education_ind],
        "address_change_ind": [address_change_ind],
        "living_status": [living_status],
        "accident_site": [accident_site],
        "past_num_of_claims": [past_num_of_claims],
        "witness_present_ind": [witness_present_ind],
        "liab_prct": [liab_prct],
        "channel": [channel],
        "policy_report_filed_ind": [policy_report_filed_ind],
        "claim_est_payout": [claim_est_payout],
        "age_of_vehicle": [age_of_vehicle],
        "vehicle_category": [vehicle_category],
        "vehicle_price": [vehicle_price],
        "vehicle_weight": [vehicle_weight],
        "latitude": [latitude],
        "longitude": [longitude]
    }

    # Create a DataFrame from the user input data
    input_df = pd.DataFrame(input_data)

    input_df = pd.DataFrame(input_data)

    X_input_encoded = column_transformer.transform(input_df)

# Use the trained model to make predictions
    best_model.set_params(**{'enable_categorical': True})
    fraud_probability = best_model.predict_proba(X_input_encoded)[:, 1]

    session['claim_number'] = claim_number
    session['fraud_probability'] = fraud_probability.tolist()

    # Display the results
    return render_template('upload_success.html',claim_number=claim_number, fraud_probability=fraud_probability)
    # return render_template('result.html', claim_number=claim_number, fraud_probability=fraud_probability)

from flask import session

@app.route('/claim', methods=['POST'])
def predict():
    claim_number = session.get('claim_number')
    fraud_probability = session.get('fraud_probability')
    print(fraud_probability)
    fraud_probability_str = ', '.join(map(str, fraud_probability))
    image_path = max(glob.glob(r'uploads\*.jpg'), key=os.path.getctime)
    with tf.Graph().as_default():
        human_string, score= prediction(image_path)
    print('model one value' + str(human_string))
    print('model one value' + str(score))
    if (human_string == 'car'):
        label_text = 'Fraud Probability of {:.2%} '.format(float(fraud_probability_str)) + ' This is not a damaged car with confidence ' + str(score) + '%. Please upload a damaged car image'
        print(image_path)
        return render_template('front.html', text = label_text, filename= image_path,claim_number=claim_number)
    elif (human_string == 'low'):
        label_text = 'Fraud Probability of {:.2%} '.format(float(fraud_probability_str)) + ' This is a low damaged car with '+ str(score) + '% confidence.'
        print(image_path)
        return render_template('front.html', text = label_text, filename= image_path,claim_number=claim_number)
    elif (human_string == 'high'):
        label_text = 'Fraud Probability of {:.2%} '.format(float(fraud_probability_str)) + ' This is a high damaged car with '+ str(score) + '% confidence.'
        print(image_path)
        return render_template('front.html', text = label_text, filename= image_path,claim_number=claim_number)
    elif (human_string == 'not'):
        label_text = 'Fraud Probability of {:.2%} '.format(float(fraud_probability_str)) + ' This is not the image of a car with confidence ' + str(score) + '%. Please upload the car image.'
        print(image_path)
        return render_template('front.html', text = label_text, filename= image_path,claim_number=claim_number)

def cleanDirectory(threadName,delay):

   while True:
       time.sleep(delay)
       print ("Cleaning Up Directory")
       filelist = [ f for f in (os.listdir(app.config['UPLOAD_FOLDER']))  ]
       for f in filelist:
         #os.remove("Uploads/"+f)
         os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

if __name__ == '__main__':
    try:
       _thread.start_new_thread( cleanDirectory, ("Cleaning Thread", 300, ) )
    except:
       print("Error: unable to start thread" )
    app.run()