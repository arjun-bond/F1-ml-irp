import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model = joblib.load("joblib/f1_model.joblib")
test_features = joblib.load("joblib/f1_test_features.joblib")
test_target = joblib.load("joblib/f1_test_target.joblib")
predictions = joblib.load('joblib/f1_predictions.joblib')
# see the importance of each feature in the decision
importance_dataframe = pd.DataFrame({
  'feature': ['GridPosition','RecentForm','Humidity','Rained','AirTemp','TrackPerformance'],
  'importance': model.feature_importances_
})

sns.barplot(data=importance_dataframe, x='feature', y='importance')
plt.show() #display bar chart with this info

cm = confusion_matrix(test_target,predictions)
sns.heatmap(cm,annot=True,fmt="d")
plt.show()