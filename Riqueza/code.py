
# coding: utf-8

# In[156]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sb


# In[157]:

train=pd.read_csv('Training_data.csv')
test=pd.read_csv('Test Data.csv')


# In[158]:

print train.head(20)
test.head()


# In[159]:

train.isnull().sum()


# In[160]:

train[train.date=='01-03-2011'].head(10)


# In[161]:

main_train=train.drop(['date', 'Serial'], axis=1)
test=test.drop('date', axis=1)
test=test.drop('Serial', axis=1)
main_train.info() 


# In[162]:

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sb.barplot(x='Hour', y='Total Visitors', data=main_train, hue='Workingday', ax=ax1)
sb.barplot(x='Weekday', y='Total Visitors', data=main_train, hue='Workingday', ax=ax2)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sb.barplot(x='time of year', y='Total Visitors', data=main_train, hue='Workingday', ax=ax1)
sb.barplot(x='Day off', y='Total Visitors', data=main_train, hue='Workingday', ax=ax2)


# In[ ]:




# In[163]:

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


# In[164]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sb.distplot(main_train['Temp'],rug=True, ax=ax1, kde=True, hist=False,)
sb.distplot(main_train['Feel_Temp'], color='red', rug=True, ax=ax1, kde=True, hist=False)
sb.distplot(main_train['Fog Density'], ax=ax2, hist=True)


# In[165]:

corr = main_train.corr()
sb.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[166]:

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


# In[176]:

from sklearn.cross_validation import train_test_split

dic=list(main_train.columns)
#stop_words=['Serial', 'Temp', ' Humidity', 'Feel_Temp', 'Foreigners', 'Local', 'Total Visitors']
stop_words=['Serial', 'temp_cat', 'Humidity_cat','Total Visitors', 'Foreigners', 'Local','Total Visitors','slot']

final_par=[ word for word in dic if word not in stop_words]

X_train,X_test, Y_train, Y_test=train_test_split(main_train[final_par], main_train['Total Visitors'], test_size=0.25, random_state=42)


# In[168]:

from sklearn.metrics import accuracy_score
import math
def costcal(prediction, Y_test, name):
    cost=(prediction-Y_test)
    cost.as_matrix()
    cost=sum(cost*cost) 
    cost=np.sqrt(cost)
    print cost
    
    print ('for {0} it is {1}'.format(name, cost))
    


# In[188]:

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(n_estimators=10).fit(X_train, Y_train)
prediction=model.predict(X_test)
(prediction)

costcal(prediction, Y_test, 'RandomForest')


# In[182]:

from sklearn import tree
model=tree.DecisionTreeRegressor().fit(X_train, Y_train)
prediction=model.predict(X_test)
costcal(prediction, Y_test, 'DecisionTree')


# In[183]:

from sklearn import svm
model=svm.SVR().fit(X_train, Y_train)
prediction=model.predict(X_test)
costcal(prediction, Y_test, 'SVM')


# In[187]:

from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor().fit(X_train, Y_train)
prediction=model.predict(X_test)
costcal(prediction, Y_test, 'GradientBoostRigressor')


# In[189]:

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor().fit(X_train, Y_train)
prediction=model.predict(X_test)
(prediction)

costcal(prediction, Y_test, 'RandomForest')
print(sum(abs(prediction-Y_test)))


# In[175]:

print X_train.columns
test.columns


# In[191]:

# predicting the test data given 

from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor().fit(main_train[final_par],main_train['Total Visitors'])
prediction=model.predict(test)
prediction
np.savetxt('results.csv', 
           np.c_[range(1,len(test)+1),prediction], 
           delimiter=',', 
           header = 'Serial,Total Visitors', 
           comments = '', 
           fmt='%d')


# In[ ]:



