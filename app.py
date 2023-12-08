from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from datetime import date
import random
from data_definitions import measure_names, installers

# Load the saved pipeline
loaded_pipeline = joblib.load('trained_pipeline.pkl')

def date_to_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", data="hey", measure_names=measure_names, installers=installers)

def preprocess_and_predict_future_data(future_data, trained_pipeline):
    # Preprocessing steps
    future_data['MeasureName'] = future_data['MeasureName'].astype('category')

    # Selecting features
    features = ['InstallationSeason', 'MeasureName', 'Installer', 'IncentiveAmount', 'InstalledMeasureQty', 'UnitOfMeasure']
    X_future = future_data[features]
    
    # Make predictions using the loaded pipeline
    y_pred_proba = trained_pipeline.predict_proba(X_future)[:, 1]

    # Add the predicted probabilities of Failing to the future data
    future_data['PredictedProbabilityFailure'] = y_pred_proba

    return y_pred_proba


@app.route("/prediction", methods=["POST"])
def prediction():
	# Extracting the date string from the form data
	date_str = request.form['InstallationDate']

	# Converting the date string to a datetime.date object
	InstallationDate = datetime.strptime(date_str, '%Y-%m-%d').date()
	MeasureName = request.form['MeasureName']
	Installer = request.form['Installer']
	IncentiveAmount = request.form['IncentiveAmount']
	InstalledMeasureQty = request.form['InstalledMeasureQty']

	# Create the test future_data DataFrame
	future_data = pd.DataFrame({
		'InstallationSeason': [date_to_season(InstallationDate)],
		'MeasureName': [MeasureName],
		'Installer': [Installer],
		'IncentiveAmount': [IncentiveAmount],
		'InstalledMeasureQty': [InstalledMeasureQty],
		'UnitOfMeasure': ["Each"]
	}, index=[0])  # Adding an index

	pred = preprocess_and_predict_future_data(future_data, loaded_pipeline)
	random_number = random.randint(1, 15)
	print("random number is: ", random_number)
	return render_template("prediction.html", data=pred, random_number=random_number)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))