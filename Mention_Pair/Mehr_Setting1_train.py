import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from joblib import dump
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import cross_validate
# Training Experiments
"""
# first grid search Experiments

dataset = pd.read_csv('full_train_fv.csv')
column_count = len(dataset.iloc[0])
x_train = dataset.iloc[:, 2:column_count - 1]
y_train = dataset.iloc[:, column_count - 1]

rfc = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators':  [200, 500, 1000],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'criterion': ['gini', 'entropy']
}

clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, scoring='f1', verbose=2)
clf.fit(x_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
"""




# second step repeated 10 fold cross validation

dataset = pd.read_csv('full_train_fv.csv')
column_count = len(dataset.iloc[0])
x_train = dataset.iloc[:, 2:column_count - 1]
y_train = dataset.iloc[:, column_count - 1]
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
model = RandomForestClassifier(n_estimators=1000, max_depth=20)
# res1 = cross_val_score(model, x_train, y_train, cv=cv, scoring='f1',verbose= 2)
# print("the average of 10 fold is\t{}".format(sum(res1)/len(res1)))
res1 = cross_validate(model, x_train, y_train, cv=cv, scoring=('precision','recall','f1' ),verbose= 2)

print("the average of precision is\t{}\nthe average of recall is\t{}\n"
      "the average of f1 is\t{}\n".format(sum(res1['test_precision'])/len(res1['test_precision']),
                                          sum(res1['test_recall']) / len(res1['test_recall']),
                                          sum(res1['test_f1']) / len(res1['test_f1'])))


# Train final optimal model and save it as file
# file_path = "./Models/Mehr_Perfect_Random_Forest"

# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(x_train, y_train)
# dump(clf, file_path + ".joblib")











