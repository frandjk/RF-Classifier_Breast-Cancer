import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('breast_cancer.csv')
df


df.shape
df.head(10)
columns = df.columns

for col in columns:
    print(df[col].value_counts)
    
df.drop('Unnamed: 32', axis = 1, inplace = True)
df

df['diagnosis'].value_counts()
df.isnull().sum()

# Declare feature vector and target variable
X = df.drop(['diagnosis '], axis = 1)
Y = df['diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size  = 0.33, random_state = 42)

X_train.shape, X_test.shape
X_train.dtypes

import category_encoders as ce 
encoder = ce.OrdinalEncoder(columns)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

X_train.head()
X_test.head()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rfc_100 = RandomForestClassifier(n_estimators = 100, random_state = 0)
rfc_100.fit(X_train, Y_train)
Y_pred_100 = rfc_100.predict(X_test)
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(Y_test, Y_pred_100)))

clf = RandomForestClassifier(n_estimators = 100, random_state = 0)
clf.fit(X_train, Y_train)
RandomForestClassifier(bootstrap = True, class_weight = None, criterion = 'gini',
                       max_depth = None, max_features = 'auto', max_leaf_nodes = None,
                       min_impurity_decrease = 0.0,
                       min_samples_leaf = 1, min_samples_split = 2,
                       min_weight_fraction_leaf = 0.0, n_estimators = 100,
                       n_jobs = None, oob_score = False, random_state = 0,
                       verbose = 0, warm_start = False 
                       )

feature_scores = pd.Series(clf.feature_importances_, index = X_train.columns).sort_values(ascending = False)
feature_scores

# visualize the feature scores
sns.barplot(x = feature_scores, y = feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.show()

