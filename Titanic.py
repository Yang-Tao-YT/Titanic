#!/usr/bin/env python
# coding: utf-8

# In[40]:


# check the documents
import os
for dirname, _, filenames in os.walk('/Users/yangtao/Downloads/titanic'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
import seaborn as sns
from sklearn.metrics import  accuracy_score 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


# In[42]:


#read  data
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
#check the shape
print(test.shape , train.shape)


# In[43]:


#check null
print(test.info())
print(train.info())


# In[44]:


# clear null
test = test.drop('Cabin',1)
test = test.fillna(test.mean().iloc[0])
train = train.drop('Cabin',1)
train = train.dropna(0,how = 'any')
train.shape


# In[45]:


# check the types of data
print(test.dtypes.unique())
col_nndata = train.select_dtypes(include = ['object']).columns
print(col_nndata)
test[col_nndata]


# In[46]:


# transfer the non numercial data
le = LabelEncoder()
test[col_nndata] = test[col_nndata].apply(lambda x:le.fit_transform(x))
train[col_nndata] = train[col_nndata].apply(lambda x:le.fit_transform(x))


# In[47]:


plt.figure(figsize=(16, 6))
sns.heatmap(train.corr(),annot = True)


# In[48]:


# clear unsignificant label
PassengerId = test.loc[:,['PassengerId']]
test = test.drop(['PassengerId','Name'],1)
train = train.drop(['PassengerId','Name'],1)

# clear outlier using k-nearest neighbors
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(train )
outlier_detector = clf.negative_outlier_factor_

train = train[outlier_detector > pd.Series(outlier_detector).quantile(0.1)]


# In[49]:


train = train[['Age','Ticket', 'Fare','Embarked', 'Sex', 'SibSp','Pclass','Parch','Survived']]
test = test[['Age','Ticket', 'Fare','Embarked', 'Sex', 'SibSp','Pclass','Parch']]


# In[50]:


#random foreset-first time try
X = train.iloc[:, :-1].values
y = train.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 42)


# In[51]:


rf_model = RandomForestClassifier().fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[52]:


# test different parameters

rf = RandomForestClassifier()
rf_params = {'random_state': [1],
             'max_depth': [10, 11, 12],
             'max_features':  ['auto', 'sqrt','og2'],
             'min_samples_leaf': [1, 2],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [113]}

grid = GridSearchCV(rf, 
                    rf_params,
                    cv = 10,   
                    n_jobs = -1)

grid.fit(X_train, y_train)


# In[56]:


#use best params
rf_adjusted = RandomForestClassifier(**grid.best_params_)


# In[57]:


cv_results = cross_val_score(rf_adjusted, X_train, y_train, cv = 10)
cv_results.mean()


# In[58]:


rf_adjusted.fit(X_train,y_train)
y_pred_adjusted = rf_adjusted.predict(X_test)
accuracy_score(y_test,y_pred_adjusted)


# In[59]:


y_pred_test = rf_adjusted.predict(test)


# In[60]:


PassengerId.loc[:,'Survived']=y_pred_test


# In[61]:


Submission = PassengerId.set_index('PassengerId')

Submission.to_csv('Submission.csv',index=True)

