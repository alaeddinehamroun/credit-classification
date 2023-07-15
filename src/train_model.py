import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score
import json
from models.random_forest import random_forest_model
from data.data_split import data_split
import os
import mlflow


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/alaeddinehamroun@gmail.com/experiment-1")
mlflow.sklearn.autolog()

# Load in the processed data
df = pd.read_csv("data/processed_data.csv")

# Split into train and test selections
x_train, x_test, y_train, y_test = data_split(df, "SeriousDlqin2yrs", split_size=0.2)

# Fit a model on the train section
model = random_forest_model()
model.fit(x_train, y_train)

# Predict
predictions = model.predict(x_test)

# Metrics
acc = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)


# Write scores to a file
with open("metrics.json", "w") as outfile:
    json.dump({"accuracy": acc, "precision": precision}, outfile)

# Plot feature importance
importances = model.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(
    list(zip(labels, importances)), columns=["feature", "importance"]
)
feature_df = feature_df.sort_values(by="importance", ascending=False)

# Image formatting
axis_fs = 18
title_fs = 22
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel("Importance", fontsize=axis_fs)
ax.set_ylabel("Feature", fontsize=axis_fs)
ax.set_title("Random forest \nfeature importance", fontsize=title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()

mlflow.end_run()
