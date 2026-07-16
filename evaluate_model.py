import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

podium_probs = joblib.load('joblib/podium_probs.joblib')
model = joblib.load("joblib/f1_model.joblib")
test_features = joblib.load("joblib/f1_test_features.joblib")
test_target = joblib.load("joblib/f1_test_target.joblib")
predictions = joblib.load('joblib/f1_predictions.joblib')

# see the importance of each feature in the decision
importance_dataframe = pd.DataFrame({
  'feature': ['GridPosition','RecentForm','Humidity','Rained','AirTemp','TrackPerformance','QualiDelta','CarPerformance'],
  'importance': model.feature_importances_
})

plt.figure(figsize=(12,6))
sns.barplot(data=importance_dataframe, x='feature', y='importance')

plt.show() #display bar chart with this info

cm = confusion_matrix(test_target,predictions)
sns.heatmap(cm,annot=True,fmt="d")
plt.show() #Display heatmap with false and true positives and negatives

# display ROC (Reciever Operating Characteristic) curve
fpr,tpr, thresholds = roc_curve(test_target, podium_probs)
roc_auc = auc(fpr,tpr)
roc_df = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
plt.figure(figsize=(12,6))
sns.lineplot(data=roc_df, x='fpr', y='tpr',label='ROC curve')
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
plt.show()