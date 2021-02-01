#!/usr/bin/env python
# coding: utf-8

# # Census income prediction 

# Predicting income based on various features to be <= $50 K or >$50 K . 

# In[108]:


# importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[109]:


original_data = pd.read_csv("censusincome.csv")
original_data


# # Data Preprocessing 

# In[110]:


original_data.head(50)


# In[111]:


# this dataset contains som values with '?' sign need to replace with nan 
original_data = original_data.replace('?',np.nan)


# In[112]:


original_data.isnull().sum()
# now we can see that workclass,occupation,native.country has sonme missing values.


# In[113]:


original_data.info()


# In[114]:


missingvalues = ['workclass','occupation','native.country']


# In[115]:


# filling the missing values 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')
original_data[missingvalues] = imputer.fit_transform(original_data[missingvalues])


# In[116]:


original_data.isnull().sum()


# In[117]:


original_data.drop(columns=['race'],axis=1,inplace=True)


# In[118]:


# for visulisation purpose cleaned data set will be used
# then after visualization respective variables will be changed to 1 and 0
# original_data.to_csv(r'M:\Study College\TY-Btech-2021\Subjects\BDAL\MP\censusincomevisuals.csv')


# In[119]:


original_data.corr()


# Categorical and Numerical Data Separation

# In[120]:


print("Numerical columns")
Numerical= original_data.select_dtypes('int64').keys()
for i in Numerical:
    print(i)
print()
print("Categorical columns")
Categorical = original_data.select_dtypes('object').keys()
for i in Categorical:
    print(i)
print()
print("Target Variable")
print("Income")


# In[121]:


# statistics for numerical columns 
original_data[Numerical].describe().T


# In[122]:


original_data.head(5)


# In[123]:


original_data.corr()


# In[124]:


original_data.select_dtypes('object').keys()


# In[125]:


# for i in original_data.select_dtypes('object').keys():
#     fig = plt.figure(figsize=(30,10))
#     sns.countplot(i, hue='income', data=original_data)
#     plt.tight_layout()
#     plt.show()


# In[126]:


original_data = original_data[original_data['native.country']=='United-States']


# In[127]:


original_data.drop(columns=['native.country','fnlwgt','education'],inplace=True)
# native.country only this dataset has significant amount of information of us
# fnlwgt has negative correlation with income
# education.num and education are same only encoding is done


# In[128]:


# original_data = pd.get_dummies(original_data,columns=["workclass", "education", "marital.status", "occupation","relationship"])
# original_data
encodings = {}

for lst in ['workclass','marital.status', 'occupation', 'relationship', 'sex', 'income'] :
    tonum = {}
    setlst = list(original_data[lst].unique())
    for i in range(len(setlst)) :
        tonum[setlst[i]] = i
    original_data[lst].replace(tonum, inplace = True)
    encodings[lst] = tonum
    
print(encodings)


# In[129]:


sns.set(rc={'figure.figsize':(13,6)}, font_scale=1.1)
sns.heatmap(original_data.corr(),annot=True,cmap='coolwarm')


# In[130]:


# original_data.drop(columns = ['marital.status','relationship'], inplace = True)


# #  Training The model

# In[131]:


original_data.head()


# In[132]:


original_data.dtypes


# In[133]:


# train test split
X = original_data.iloc[:, :-1]
Y = original_data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[134]:


# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

predict_dtc = dtc.predict(X_test)

print('Decision Tree Classifier :', accuracy_score(Y_test, predict_dtc) * 100)


# In[135]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)

predict_lr = lr.predict(X_test)

print('Logistic Regression :', accuracy_score(Y_test, predict_lr) * 100)


# In[136]:


rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(X_train, Y_train)

predict_rfc= rfc.predict(X_test)

print('Random Forest Classifier :', accuracy_score(Y_test, predict_rfc) * 100)


# In[137]:


# training accuracies
# for DTC
predict_train_dtc = dtc.predict(X_train)
train_acc_dtc = accuracy_score(Y_train,predict_train_dtc)
print("Training Accuracy For Decision Tree : " , train_acc_dtc*100)
# for lr
predict_train_lr = lr.predict(X_train)
train_acc_lr = accuracy_score(Y_train,predict_train_lr)
print("Training Accuracy For Logistic Regression : " ,train_acc_lr*100)
# for rfc
predict_train_rfc = rfc.predict(X_train)
train_acc_rfc = accuracy_score(Y_train,predict_train_rfc)
print("Training Accuracy For Random Forest Classifier : " , train_acc_rfc*100)


# In[138]:


results = {'Method':['Decision Tree Classifier','Logistic Regression','Random Forest Classifier'],
           'Training Accuracy':[train_acc_dtc*100,train_acc_lr*100,train_acc_rfc*100],
           'Testing Accuracy Score':[accuracy_score(Y_test, predict_dtc) * 100,
                                     accuracy_score(Y_test, predict_lr) * 100,
                                    accuracy_score(Y_test, predict_rfc) * 100]}
final_view = pd.DataFrame(results)
final_view


# In[139]:


import pickle

f = open('store.pckl', 'wb')
pickle.dump({'rfc' : rfc, 'encodings' : encodings}, f)
f.close()


# In[ ]:




