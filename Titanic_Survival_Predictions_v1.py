#Titanic_Survival_Predictions_1.py

"""
Random Forest Classifier ML model predicts survival based on passenger demographics.
    Variables= gender, passenger class, family members onboard
Training data and testing data provided by Kaggle competition; provided
    /kaggle/input/titanic/train.csv
    /kaggle/input/titanic/test.csv
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 

#id input data for training and testing.  uses working directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


#Use scikit-learn random forest classifier algorithm to build decision tree model from training data.
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

#Use test data to form predictions. Output 2 columns to csv for submission
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
