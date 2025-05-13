import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Placement_Prediction_data.csv')

# Remove unnamed columns (typically from index column)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handle missing values
df.fillna(0, inplace=True)

# Define features and target
x = df.drop(['StudentId', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Encode categorical columns
le = preprocessing.LabelEncoder()
x['Internship'] = le.fit_transform(x['Internship'])
x['Hackathon'] = le.fit_transform(x['Hackathon'])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Train model
classify = RandomForestClassifier(n_estimators=100, criterion="entropy")
classify.fit(x_train, y_train)

# Predict and evaluate
ypred = classify.predict(x_test)
accuracy = accuracy_score(y_test, ypred)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(classify, open('model.pkl', 'wb'))

# Load model and predict on sample input
model = pickle.load(open('model.pkl', 'rb'))

# Check expected input columns
print("Input Features:", x.columns.tolist())

# Predict using sample input (must match column order exactly)
sample_input = [[8, 1, 3, 2, 9, 4.8, 0, 1, 71, 87, 0]] 
print("Placement Prediction:", model.predict(sample_input))
