import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

Dataset = pd.read_csv('f1_data_sorted.csv')
#  & (Dataset['RoundNumber'] <=11)
print("Splitting dataset")
train_data = Dataset[(Dataset['Year'] == 2023) |(Dataset['Year'] == 2022) | ((Dataset['Year'] == 2024))] #use 2022 and half 2023 season as training data
test_data = Dataset[(Dataset['Year'] == 2025)] # use half 2023 season as test data

# Split training data
train_target = train_data['Podium']
train_features = train_data[['GridPosition','RecentForm','Humidity','Rained','AirTemp','TrackPerformance','QualiDelta','CarPerformance']]

#split test data
test_target = test_data['Podium']
test_features = test_data[['GridPosition','RecentForm','Humidity','Rained','AirTemp','TrackPerformance','QualiDelta','CarPerformance']]


print("Training model")
# init the model then supply the data
model = RandomForestClassifier()
model.fit(train_features, train_target)
predictions = model.predict(test_features) #predict outcomes on test data


score = accuracy_score(test_target, predictions) # calculate an accuracy score

# Predict probabilities for each driver
probability = model.predict_proba(test_features)
podium_probs = probability[:, 1] # Select podium probability

joblib.dump(podium_probs,'joblib/podium_probs.joblib')
joblib.dump(model, 'joblib/f1_model.joblib') #save model
joblib.dump(test_features, 'joblib/f1_test_features.joblib')
joblib.dump(predictions, 'joblib/f1_predictions.joblib')
joblib.dump(test_target, 'joblib/f1_test_target.joblib')