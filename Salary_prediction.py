import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('Salary_prediction_data.csv')  
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.fillna(0, inplace=True)

# Feature/Target separation
x = df.drop(['StudentId', 'salary'], axis=1)
y = df['salary']

# Label encode categorical fields
le = preprocessing.LabelEncoder()
x['Internship'] = le.fit_transform(x['Internship'])
x['Hackathon'] = le.fit_transform(x['Hackathon'])
x['PlacementStatus'] = le.fit_transform(x['PlacementStatus'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Train model
classify = RandomForestClassifier(n_estimators=100, criterion="entropy")
classify.fit(x_train, y_train)

# Predict and evaluate
ypred = classify.predict(x_test)
print("Model Accuracy:", accuracy_score(ypred, y_test))

# Save model
pickle.dump(classify, open('model1.pkl', 'wb'))

# Load model and make a sample prediction
model1 = pickle.load(open('model1.pkl', 'rb'))
sample_input = [[8, 1, 3, 2, 9, 4.8, 0, 1, 71, 87, 0, 1]]
print("Predicted Salary:", model1.predict(sample_input))
