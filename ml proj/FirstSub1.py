# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:36:18 2021

@author: Sourav
"""
from sklearn.metrics import accuracy_score
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('trainRandom.csv')
test = pd.read_csv('test.csv')
test = test.drop(['cont0','cont1',], axis=1)
train = train.drop(['id','cont0','cont1'], axis=1)
X = train.drop('target', axis=1)
y = train.target
print(y)
train = train.drop(['target'], axis=1)
print(train)
"""def describe_data(df):
    print("Data Types:")
    print(df.dtypes)
    print("Rows and Columns:")
    print(df.shape)
    print("Column Names:")
    print(df.columns)
    print("Null Values:")
    print(df.apply(lambda x: sum(x.isnull()) / len(df)))
describe_data(train)"""

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).columns
"""cf=pd.DataFrame(numeric_features)
nf=pd.DataFrame(categorical_features)
cf=cf.drop(['target'])"""
#print(categorical_features)
#print(numeric_features)
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('label',OneHotEncoder(handle_unknown='ignore'))])
#print(numeric_transformer)
#print(categorical_transformer)
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])
#print(preprocessor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state = 0)
#lr = Pipeline(steps=[('preprocessor', preprocessor),('regressor', RandomForestRegressor(random_state=1))])
from sklearn.linear_model import LogisticRegression
lr = Pipeline(steps=[('preprocessor', preprocessor),('classifier',LinearSVC())])
lr.fit(X_train, y_train)
pickle.dump(lr, open('model.pkl','wb'))
melb_preds = lr.predict(X_test)
#predictions = [round(value) for value in melb_preds]
print(melb_preds)
#print(mean_absolute_error(y_test, predictions))
print(mean_absolute_error(y_test, melb_preds))
#print("model score: %.3f" % lr.score(X_test, y_test))
#accuracy = accuracy_score(y_test, melb_preds)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
test_no_id = test.drop('id', axis=1)
test_predictions = lr.predict(test_no_id)
predictions2 = [round(value) for value in test_predictions]
id = test['id']
submission_df_1 = pd.DataFrame({"id": id,"target":test_predictions})
submission_df_1.to_csv('OnePipeline212.csv', index=False)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, melb_preds))




#print(X_train)
#print(y_train)
#lr.fit(X_train, y_train)
#lr.fit(X_train, y_train)
"""melb_preds = lr.predict(X_test)
print(melb_preds)
print(mean_absolute_error(y_test, melb_preds))"""
#print("model score: %.3f" % lr.score(X_test, y_test))



#('classifier', RandomForestRegressor(n_estimators=5, random_state=0,max_depth=5)) use this with rando forest regress














