# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:42:59 2021

@author: Sourav
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train.head(15))
#print(train.describe())
#print(train.LotFrontage.isnull().sum().max())
#print(train.columns)
train=train.fillna(train.mean())
test=test.fillna(train.mean())
#print(train.isnull().sum().max())
print("Null Values:")
print(train.apply(lambda x: sum(x.isnull()) / len(train)))
#train.apply(lambda x: x.fillna(x.mean()),axis=0)
#print(train.LotFrontage.isnull().sum().max())
#print(train['Revenue'].describe())
#print(train['Revenue'].unique())
"""def describe_data(train):
    print("Data Types:")
    print(train.dtypes)
    print("Rows and Columns:")
    print(train.shape)
    print("Column Names:")
    print(train.columns)
    print("Null Values:")
    print(train.apply(lambda x: sum(x.isnull()) / len(train)))
describe_data(train)"""
#yrsold,miscval,enclosed porch,basement half bath,lowqualfinalsf,bsmtfinsf2,overallcond,msubclass,id1
train = train.drop(['Id','YrSold','MiscVal','EnclosedPorch','BsmtHalfBath','BsmtFinSF2','LowQualFinSF','OverallCond','MSSubClass','KitchenAbvGr','Alley'], axis=1)
#test = test.drop(['YrSold','MiscVal','EnclosedPorch','BsmtHalfBath','BsmtFinSF2','LowQualFinSF','OverallCond','MSSubClass','KitchenAbvGr',], axis=1)

#corrmat = train.corr()
#f,ax = plt.subplots(figsize=(20,20))
#sns.heatmap(corrmat,vmax=0.8,square=True,annot=True,annot_kws={'size':8})
print("helloooooooooooooooooo")
obj_df = train.select_dtypes(include=['object']).copy()
#print(obj_df.head())
#from sklearn.preprocessing import OrdinalEncoder

le1=LabelEncoder()
for i in obj_df:
   train[i]=le1.fit_transform(train[i])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
#train = scaler.fit_transform(train)
   
#obj_df1 = test.select_dtypes(include=['object']).copy()
#print(obj_df.head())
#te1=LabelEncoder()
#for i in obj_df:
   #test[i]=te1.fit_transform(test[i])
#print(train.head())
   
x=train.drop(['SalePrice'],axis=1)
y=train.SalePrice

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=7)
print("shape of X Train :"+str(X_train.shape))
print("shape of X Test :"+str(X_test.shape))
print("shape of Y Train :"+str(Y_train.shape))
print("shape of Y Test :"+str(Y_test.shape))
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
"""for this_C in [1,2,3,4,5,6,7,8,9,10]:
    model1 = GradientBoostingRegressor(n_estimators=110,random_state=1,learning_rate=0.2,max_depth=4)
    model1.fit(X_train,Y_train)
    scoretrain = model1.score(X_train,Y_train)
    scoretest  = model1.score(X_test,Y_test)
    print("random forest regressor value of n_estimators:{}, training score :{:2f} , Test Score: {:2f} \n".format(this_C,scoretrain,scoretest))"""
model1 = RandomForestRegressor(n_estimators=99,random_state=2)
model1.fit(X_train,Y_train)
y_pred = model1.predict(X_test)
from sklearn.metrics import mean_squared_error
rms = mean_squared_error(Y_test, y_pred, squared=False)
print (rms)
#print(y_pred)
#print(mean_absolute_error(Y_test,y_pred))
from sklearn.model_selection import RepeatedKFold

from numpy import mean
from numpy import std
model = GradientBoostingRegressor(n_estimators=125,random_state=7,learning_rate=0.22,max_depth=4)
# define the evaluation procedure
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
#n_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
model.fit(X_train,Y_train)
y_pred1 = model.predict(X_test)
#print(y_pred)
# report performance
#print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#print(accuracy_score(Y_test,y_pred))
#print(precision_score(Y_test,y_pred))


rms = mean_squared_error(Y_test, y_pred1, squared=False)
print(rms)
rmse = np.sqrt(np.mean(((y_pred1 - Y_test) ** 2)))
print(rmse)
#print(mean_absolute_error(Y_test,y_pred1))
from keras.models import Sequential
from keras.layers import Dense, LSTM
#import matplotlib.pyplot as plt
#import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

"""X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(128, return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, Y_train, batch_size=1, epochs=0)
predictions = model.predict(X_test)
rmse = np.sqrt(np.mean(((predictions - Y_test) ** 2)))
print(rmse)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - Y_test) ** 2)))
print(rmse)"""


"""test_no_id = test.drop('Id', axis=1)
test_predictions = model1.predict(test_no_id)
id = test['Id']
submission_df_1 = pd.DataFrame({"Id": id,"SalePrice":test_predictions})
submission_df_1.to_csv('Housepredeval.csv', index=False)"""








#RMSE 25757.39497686014
#RMSE 24284.056917482743
#RMSE 24872.399960
# 24648.49

