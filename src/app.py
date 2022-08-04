import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import pickle
import seaborn as sns
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')
df_raw

df_raw.info()
df_raw.describe()
df_raw.shape
df_raw.describe(include='object')
df_raw.sample(20)

# duplicated
for col in df_raw.columns:
    duplicates = df_raw[col].duplicated().sum()
    print(f'Dataset have {duplicates} duplicated {df_raw[col].name}')


df = df_raw.copy().drop(['PassengerId','Cabin', 'Ticket', 'Name'], axis=1)
df

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_train.sample(10)

df_train.info()
df_train.sample(10)
df_train.describe()
df_train.describe(include='object')


X_train.hist(figsize=(8,8), alpha=0.5)
plt.show()

fig, ax = plt.subplots(1,5, figsize=(10,4))
axs = ax.flatten()
for i in range(len(axs)):
    X_train._get_numeric_data().iloc[:,i].to_frame().boxplot(ax=axs[i])

pd.plotting.scatter_matrix(X_train, diagonal='kde', figsize=(8,8), c='Violet')
plt.show()

sns.pairplot(data=X_train, hue='Sex')
plt.show()

sns.pairplot(data=X_train, hue='Embarked')
plt.show()

columns = ['Sex', 'Embarked']

for col in columns:
    sns.countplot(data=X_train, x=col)
    plt.show()

sns.countplot(data=X_train, x='Sex', hue='Embarked')
plt.show()

X_train.corr().style.background_gradient(cmap='Blues')
df_train.corr().style.background_gradient(cmap='Blues')

fig, axs = plt.subplots(ncols=3, figsize=(10,10))
sns.boxplot(data=df_train, x='Survived', y='Age', ax=axs[0])
sns.boxplot(data=df_train, x='Survived', y='Fare', ax=axs[1])
sns.boxplot(data=df_train, x='Survived', y='Parch', ax=axs[2])
plt.show()

imputer_mean = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer_mean = imputer_mean.fit(X_train[['Age']])
X_train['Age'] = imputer_mean.transform(X_train[['Age']])
print('Age mean is:',imputer_mean.statistics_)

imputer_mode = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer_mode = imputer_mode.fit(X_train[['Embarked']])
X_train['Embarked'] = imputer_mode.transform(X_train[['Embarked']])
print('Embarked mode is:',imputer_mode.statistics_)

X_test['Age'] = imputer_mean.transform(X_test[['Age']])
X_test['Embarked'] = imputer_mode.transform(X_test[['Embarked']])

X_train[['Sex','Embarked']] = X_train[['Sex','Embarked']].astype('category')
X_test[['Sex','Embarked']] = X_test[['Sex','Embarked']].astype('category')

X_train['Sex'] = X_train['Sex'].cat.codes
X_train['Embarked'] = X_train['Embarked'].cat.codes

X_test['Sex'] = X_test['Sex'].cat.codes
X_test['Embarked'] = X_test['Embarked'].cat.codes

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
print(f'Accuracy in train dataset: {rfc.score(X_train, y_train)}')
print(f'Accuracy in test dataset: {rfc.score(X_test, y_test)}')

y_pred_rfc = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rfc, labels=rfc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=rfc.classes_)
disp.plot()
plt.show()

print(classification_report(y_test,y_pred_rfc))

param_grid = [{
    'max_depth': [8, 12, 16], 
    'min_samples_split': [12, 16, 20], 
    'criterion': ['gini', 'entropy']
}]
param_grid

rfc2 = RandomForestClassifier(random_state=1107)
grid =  GridSearchCV(estimator=rfc2, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
print('Best parameters to select:', grid.best_params_)

rfc_model = grid.best_estimator_
y_pred_cv = rfc_model.predict(X_test)
print(f'Accuracy RFC selected by CV is {grid.score(X_test, y_test)}')

cm = confusion_matrix(y_test, y_pred_cv, labels=grid.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=grid.classes_)
disp.plot()
plt.show()
print(classification_report(y_test,y_pred_cv))

importance = rfc_model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

plt.figure(figsize=(30,5))
columns = X.columns
sns.barplot(columns, importance)
plt.title('Feature importance')
plt.show()
columns

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
criterion=['gini','entropy']

random_grid = {'n_estimators': n_estimators,
#'max_features': max_features, # Son muy pocas variables por lo cual no vale la pena aplicarlo
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap,
'criterion':criterion}

print(random_grid)

rfc3 = RandomForestClassifier(random_state=1107)

grid_random = RandomizedSearchCV(
    estimator=rfc3,
    n_iter=100,
    cv=5,
    random_state=1107,
    param_distributions=random_grid)

grid_random.fit(X_train,y_train)
print('Best parameters:', grid_random.best_params_)

best_param = grid_random.best_params_
best_model = RandomForestClassifier(**best_param)
best_model
print('Best model:', grid_random.best_estimator_) 

model_cv_2 = grid_random.best_estimator_
importance = model_cv_2.feature_importances_

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

plt.figure(figsize=(30,5))
columns = X_train.columns
sns.barplot(columns, importance)
plt.title('Feature importance')
plt.show()

filename = '../models/final_model.sav'
pickle.dump(model_cv_2, open(filename, 'wb'))

# # XGB
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
print('Accuracy of xgboost is:',accuracy_score(y_test, pred))

xgb_2 = XGBClassifier()

parameters = {
    "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
    "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight" : [ 1, 3, 5, 7 ],
    "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
}

grid_xgb = RandomizedSearchCV(
    xgb_2,
    parameters, n_jobs=4,
    scoring="neg_log_loss",
    cv=3)

grid_xgb.fit(X_train, y_train)

xgb_2 = grid_xgb.best_estimator_
prediction = xgb_2.predict(X_test)
print(f'Accuracy xgboost with hyperparameters: {accuracy_score(y_test, prediction)}')

