import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('Placement_Prediction_data.csv')

# Remove any unnamed columns (often index columns)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Fill missing values with 0
df.fillna(0, inplace=True)

# Separate features and target
X = df.drop(['StudentId', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Label encode categorical columns
le = LabelEncoder()
X['Internship'] = le.fit_transform(X['Internship'])
X['Hackathon'] = le.fit_transform(X['Hackathon'])

# Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Train the model
model = RandomForestClassifier(n_estimators=100, criterion="entropy")
model.fit(x_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))

# Test a prediction
loaded_model = pickle.load(open('model.pkl', 'rb'))
sample_input = [[8, 1, 3, 2, 9, 4.8, 0, 1, 71, 87, 0]]
prediction = loaded_model.predict(sample_input)
print(f"Prediction for sample input: {prediction[0]}")
