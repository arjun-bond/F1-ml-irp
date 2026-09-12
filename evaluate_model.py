import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score

# Scenario names match the suffixes used when saving joblib files in train_model.py
scenarios = ['2022', '2021']

# Compare the model against two baselines across each training-window scenario:
# - Naive: assumes the top 3 starting grid positions end up on the podium
# - Historical: assumes the top 3 in the drivers' championship standings (before the race) end up on the podium
comparison_rows = []
for name in scenarios:
    test_target = joblib.load(f'joblib/f1_test_target_{name}.joblib')
    predictions = joblib.load(f'joblib/f1_predictions_{name}.joblib')
    test_data = joblib.load(f'joblib/f1_test_data_{name}.joblib')

    rain_mask = test_data['Rained'].astype(bool).to_numpy()
    naive_predictions = (test_data['GridPosition'] <= 3).astype(int).to_numpy()
    historical_predictions = (test_data['ChampionshipRank'] <= 3).astype(int).to_numpy()

    scenario_label = f"Train {name}"
    model_predictions = {
        'RandomForest': predictions,
        'Naive (Grid Top3)': naive_predictions,
        'Historical (Standings Top3)': historical_predictions,
    }

    for model_name, model_preds in model_predictions.items():
        dry_accuracy = accuracy_score(test_target[~rain_mask], model_preds[~rain_mask])
        wet_accuracy = accuracy_score(test_target[rain_mask], model_preds[rain_mask])
        overall_accuracy = accuracy_score(test_target, model_preds)

        comparison_rows.append({'scenario': scenario_label, 'model': model_name, 'condition': 'Dry', 'accuracy': dry_accuracy, 'n': int((~rain_mask).sum())})
        comparison_rows.append({'scenario': scenario_label, 'model': model_name, 'condition': 'Wet', 'accuracy': wet_accuracy, 'n': int(rain_mask.sum())})
        comparison_rows.append({'scenario': scenario_label, 'model': model_name, 'condition': 'Overall', 'accuracy': overall_accuracy, 'n': len(test_target)})

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df)

# Overall accuracy: model vs baselines, per scenario
plt.figure(figsize=(10,6))
overall_df = comparison_df[comparison_df['condition'] == 'Overall']
sns.barplot(data=overall_df, x='scenario', y='accuracy', hue='model')
plt.ylim(0, 1)
plt.title('Model vs Baselines: Overall Accuracy')
plt.show()

# Dry vs wet accuracy, faceted by model
dry_wet_df = comparison_df[comparison_df['condition'] != 'Overall']
g = sns.catplot(data=dry_wet_df, x='scenario', y='accuracy', hue='condition', col='model', kind='bar')
g.set(ylim=(0, 1))
g.figure.suptitle('Model vs Baselines: Dry vs Wet Accuracy', y=1.05)
plt.show()

# Detailed diagnostic plots for each scenario: feature importance, confusion matrix, ROC, PRC
for name in scenarios:
    podium_probs = joblib.load(f'joblib/podium_probs_{name}.joblib')
    model = joblib.load(f'joblib/f1_model_{name}.joblib')
    test_features = joblib.load(f'joblib/f1_test_features_{name}.joblib')
    test_target = joblib.load(f'joblib/f1_test_target_{name}.joblib')
    predictions = joblib.load(f'joblib/f1_predictions_{name}.joblib')
    test_data = joblib.load(f'joblib/f1_test_data_{name}.joblib')

    # see the importance of each feature in the decision
    importance_dataframe = pd.DataFrame({
      'feature': ['GridPosition','RecentForm','Humidity','Rained','AirTemp','TrackPerformance','QualiDelta','CarPerformance'],
      'importance': model.feature_importances_
    })

    plt.figure(figsize=(12,6))
    sns.barplot(data=importance_dataframe, x='feature', y='importance')
    plt.title(f'Feature Importance (Scenario: Train {name})')
    plt.show() #display bar chart with this info

    cm = confusion_matrix(test_target,test_data['Top3_Prediction'])
    sns.heatmap(cm,annot=True,fmt="d")
    plt.title(f'Confusion Matrix (Scenario: Train {name})')
    plt.show() #Display heatmap with false and true positives and negatives

    # display ROC (Reciever Operating Characteristic) curve
    fpr,tpr, thresholds = roc_curve(test_target, podium_probs)
    roc_auc = auc(fpr,tpr)
    roc_df = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
    plt.figure(figsize=(12,6))
    sns.lineplot(data=roc_df, x='fpr', y='tpr',label='ROC curve')
    plt.title(f'ROC Curve (Scenario: Train {name})')
    plt.text(
      0.8,
      0.1,
      "ROC score: %.2f" % roc_auc,
      bbox={'facecolor':'white', 'edgecolor':'black', 'alpha':0.5}
    )
    plt.show()

    # Display Precision Recall Curve (PRC)
    precision, recall, thresholds = precision_recall_curve(test_target, podium_probs)
    precision = precision[:-1]
    recall = recall[:-1]
    plt.figure(figsize=(12,6))
    prc_df = pd.DataFrame({'precision':precision,'recall':recall,'threshold':thresholds})
    sns.lineplot(data=prc_df, x='threshold', y='precision',label='Precision')
    sns.lineplot(data=prc_df, x='threshold', y='recall',label='Recall')
    plt.title(f'Precision/Recall Curve (Scenario: Train {name})')
    plt.show()