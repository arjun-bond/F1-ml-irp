import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

Dataset = pd.read_csv('f1_data_sorted.csv')

feature_cols = ['GridPosition','RecentForm','Humidity','Rained','AirTemp','TrackPerformance','QualiDelta','CarPerformance']

# Each scenario trains on a window of seasons and predicts the following season
scenarios = {
    '2022': {'train_years': (2019, 2021), 'test_year': 2022},
    '2021': {'train_years': (2018, 2020), 'test_year': 2021},
}

for name, cfg in scenarios.items():
    print(f"Splitting dataset for scenario '{name}'")
    train_data = Dataset[Dataset['Year'].between(*cfg['train_years'])]
    test_data = Dataset[Dataset['Year'] == cfg['test_year']].copy()

    train_target = train_data['Podium']
    train_features = train_data[feature_cols]

    test_target = test_data['Podium']
    test_features = test_data[feature_cols]

    print(f"Training model for scenario '{name}'")
    model = RandomForestClassifier()
    model.fit(train_features, train_target)
    predictions = model.predict(test_features) #predict outcomes on test data

    score = accuracy_score(test_target, predictions) # calculate an accuracy score
    print(f"Scenario '{name}' accuracy: {score:.3f}")

    # Predict probabilities for each driver
    probability = model.predict_proba(test_features)
    podium_probs = probability[:, 1] # Select podium probability

    test_data['Podium_probs'] = podium_probs
    test_data['Race_Rank'] = test_data.groupby(['Year', 'RoundNumber'])['Podium_probs'].rank(ascending=False, method='first')
    test_data['Top3_Prediction'] = (test_data['Race_Rank'] <= 3).astype(int)

    joblib.dump(podium_probs, f'joblib/podium_probs_{name}.joblib')
    joblib.dump(model, f'joblib/f1_model_{name}.joblib')  # save model
    joblib.dump(test_features, f'joblib/f1_test_features_{name}.joblib')
    joblib.dump(predictions, f'joblib/f1_predictions_{name}.joblib')
    joblib.dump(test_target, f'joblib/f1_test_target_{name}.joblib')
    joblib.dump(test_data, f'joblib/f1_test_data_{name}.joblib')