import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

Dataset = pd.read_csv('f1_data_sorted.csv')


train_data = Dataset[(Dataset['Year'] == 2022) | ((Dataset['Year'] == 2023) & (Dataset['RoundNumber'] <=11))] #use 2022 and half 2023 season as training data
test_data = Dataset[(Dataset['Year'] == 2023) & (Dataset['RoundNumber'] > 11)] # use half 2023 season as test data

# Split training data
train_target = train_data['Podium']
train_features = train_data[['GridPosition','RecentForm']]

#split test data
test_target = test_data['Podium']
test_features = test_data[['GridPosition','RecentForm']]

# init the model then supply the data
model = RandomForestClassifier()
model.fit(train_features, train_target)
predictions = model.predict(test_features) #predict outcomes on test data


score = accuracy_score(test_target, predictions) # calculate an accuracy score
