import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime

# Function to convert dates to seasons
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

# Loading the dataset
file_path = 'wfs_master_data_v2.csv'
data = pd.read_csv(file_path)

# Preprocessing steps
# 1. Combining FailQty and FailCodes to create the target variable
data['FailedInspection'] = np.where((data['FailedQty'] > 0) | (data['FailCodes'].notna()), 1, 0)

# 2. Converting 'MeasureName' to a categorical feature
data['MeasureName'] = data['MeasureName'].astype('category')

# 3. Converting 'InstallationDate' to a season variable
data['InstallationDate'] = pd.to_datetime(data['InstallationDate'], errors='coerce')
data['InstallationSeason'] = data['InstallationDate'].apply(date_to_season)

# 4. Selecting final features
features = ['InstallationSeason', 'MeasureName', 'Installer', 'IncentiveAmount', 'InstalledMeasureQty', 'UnitOfMeasure']
X = data[features]
y = data['FailedInspection']

# Preprocessing Pipeline
categorical_features = ['InstallationSeason', 'MeasureName', 'Installer', 'UnitOfMeasure']
numerical_features = ['IncentiveAmount', 'InstalledMeasureQty']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
pipeline.fit(X_train, y_train)

import joblib
# Serialize and save the trained pipeline
joblib.dump(pipeline, 'trained_pipeline.pkl')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')

# Calculate recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.2f}')

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.2f}')

# Calculate ROC AUC
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC: {roc_auc:.2f}')

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Now you can plot the ROC curve if desired
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()