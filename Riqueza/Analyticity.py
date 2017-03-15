
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sb


# In[3]:

train=pd.read_csv('Training_data.csv')


# In[4]:

train.head(20)


# In[5]:

train.isnull().sum()


# In[32]:

train[train.date=='01-03-2011'].head(10)


# In[7]:

main_train=train.drop('date', axis=1)
temp=train.drop('dat')
main_train.info()


# In[8]:

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sb.barplot(x='Hour', y='Total Visitors', data=main_train, hue='Workingday', ax=ax1)
sb.barplot(x='Weekday', y='Total Visitors', data=main_train, hue='Workingday', ax=ax2)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sb.barplot(x='time of year', y='Total Visitors', data=main_train, hue='Workingday', ax=ax1)
sb.barplot(x='Day off', y='Total Visitors', data=main_train, hue='Workingday', ax=ax2)


# In[ ]:




# In[50]:

main_train['slot']=0
main_train.loc[main_train['Hour']<7, 'slot']=1
main_train.loc[(main_train['Hour']>6) & (main_train['Hour']<11), 'slot']=2
main_train.loc[(main_train['Hour']>10) & (main_train['Hour']<17), 'slot']=3
main_train.loc[(main_train['Hour']>16) & (main_train['Hour']<21), 'slot']=4
main_train.loc[(main_train['Hour']>20) & (main_train['Hour']<25), 'slot']=4

sb.boxplot(main_train[' Humidity'],)
main_train=main_train.loc[main_train[' Humidity']>-100]
main_train=main_train.loc[main_train['Fog Density']>0.03]

fig, (ax1, ax2)=plt.subplots(1,2)
sb.boxplot(main_train[' Humidity'], ax=ax1)
sb.boxplot(main_train['Fog Density'], ax=ax2)


# In[15]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sb.distplot(main_train['Temp'],rug=True, ax=ax1, kde=True, hist=False,)
sb.distplot(main_train['Feel_Temp'], color='red', rug=True, ax=ax1, kde=True, hist=False)
sb.distplot(main_train['Fog Density'], ax=ax2, hist=True)


# In[16]:

corr = main_train.corr()
sb.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[48]:

main_train['temp_cat']=0
main_train['Humidity_cat']=0
'''def to_cat(x,y,z):
    c=0   
    for i in xrange(10):
        x.loc[(y>i/10)&(y<i/10), z]=i
'''
def to_cat(x, y):
    c=0   
    for i in xrange(-1,25):
        main_train.loc[(x>float(i)/float(10))&(x<float(i+1)/float(10)), y]=i

to_cat(main_train['Feel_Temp'], 'temp_cat')
to_cat(main_train[' Humidity'], 'Humidity_cat')
main_train.head(10)


# In[138]:

from sklearn.cross_validation import train_test_split

dic=list(main_train.columns)
#stop_words=['Serial', 'Temp', ' Humidity', 'Feel_Temp', 'Foreigners', 'Local', 'Total Visitors']
stop_words=['Serial', 'temp_cat', 'Humidity_cat', 'Feel_Temp','Total Visitors']

final_par=[ word for word in dic if word not in stop_words]

X_train,X_test, Y_train, Y_test=train_test_split(main_train[final_par], main_train['Total Visitors'], test_size=0.25, random_state=42)


# In[149]:

from sklearn.metrics import accuracy_score
import math
def costcal(prediction, Y_test, name):
    cost=(prediction-Y_test)
    cost.as_matrix()
    cost=sum(cost*cost) 
    cost=np.sqrt(cost)
    print cost
    
    print ('for {0} it is {1}'.format(name, cost))
    


# In[151]:

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor().fit(X_train, Y_train)
prediction=model.predict(X_test)
(prediction)

costcal(prediction, Y_test, 'RandomForest')


# In[150]:

from sklearn import tree
model=tree.DecisionTreeRegressor().fit(X_train, Y_train)
prediction=model.predict(X_test)
costcal(prediction, Y_test, 'DecisionTree')


# In[144]:

from sklearn import svm
model=svm.SVR().fit(X_train, Y_train)
prediction=model.predict(X_test)
costcal(prediction, Y_test, 'SVM')


# In[145]:

from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor().fit(X_train, Y_train)
prediction=model.predict(X_test)
costcal(prediction, Y_test, 'SVM')


# In[ ]:

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor().fit(X_train, Y_train)
prediction=model.predict(X_test)
(prediction)

costcal(prediction, Y_test, 'RandomForest')

