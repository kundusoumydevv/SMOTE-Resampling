#!/usr/bin/env python
# coding: utf-8

# In[265]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[266]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[267]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[268]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


# In[269]:


df = pd.read_csv("data.txt")


# In[270]:


df.head()


# In[271]:


df.tail()


# In[272]:


df.info()


# In[273]:


df[df.columns].hist(bins=50, figsize=(20,15), color='#5486d6')
plt.show


# In[274]:


sns.countplot(x="Marital status" , data = df)


# In[275]:


df['Marital status'].value_counts()


# In[276]:


X = df.iloc[:, 1:4]
X


# In[277]:


Y = df.iloc[:, 0:1]
Y


# In[278]:


from sklearn.model_selection import train_test_split

X_train, X_test,Y_train, Y_test =train_test_split(X, Y, test_size = 0.3, random_state=0)


# In[279]:


df['Marital status'].value_counts()


# In[280]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
# resampling X, y
X_ros, y_ros = ros.fit_resample(X, Y)


# In[281]:


y_ros.value_counts()


# In[282]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler() 
# resampling X, y
X_rus, y_rus = rus.fit_resample(X, Y)


# In[283]:


y_rus.value_counts()


# In[284]:


# instantiating over and under sampler
over = RandomOverSampler(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.8)


# In[285]:


# first performing oversampling to minority class


from imblearn.over_sampling import SMOTE
#oversample= SMOTE()
#X_over, y_over = oversample.fit_resample(X,Y)



X_over, y_over = SMOTE(k_neighbors=3).fit_resample(X, Y)


# In[286]:


y_over.value_counts()


# In[287]:


#X_combined_sampling, y_combined_sampling = under.fit_resample(X_over , y_over)


# In[288]:


from imblearn.over_sampling import SMOTE


# In[289]:


SMOTE=SMOTE()


# In[300]:


from imblearn.over_sampling import SMOTE
X_train_SMOTE, X_test_SMOTE,Y_train_SMOTE, Y_test_SMOTE =train_test_split(X, Y, test_size = 0.3, random_state=0)
X_train_SMOTE, Y_train_SMOTE = SMOTE(k_neighbors=1).fit_resample(X_train , Y_train)
Y_train_SMOTE.value_counts()


# In[301]:


Y_train_SMOTE.value_counts()


# # Logistic Regression On Normal Split
# 
# 

# In[292]:


from sklearn.linear_model import LogisticRegression


# In[293]:


logmodel = LogisticRegression(max_iter=10000)


# In[294]:


logmodel.fit(X_train,Y_train)


# In[297]:


predictions = logmodel.predict(X_test)


# In[298]:


print("Accuracy: ", accuracy_score(Y_test,predictions))

#print("F1 score: ", f1_score(y_test,predictions,pos_label='positive',average='micro'))
#print("Recall: ", recall_score(y_test,predictions,pos_label='positive',average='micro'))
#print("Precision: ", precision_score(y_test,predictions,pos_label='positive',average='micro'))

print("\n")
print(classification_report(Y_test,predictions))
print("\n")
print(confusion_matrix(Y_test,predictions))


# # Logistic Regression on SMOTE Resampling

# In[302]:


logmodel2 = LogisticRegression(max_iter = 10000)


# In[305]:


logmodel2.fit(X_train_SMOTE ,Y_train_SMOTE )


# In[306]:


pridiction2 = logmodel.predict(X_test_SMOTE)


# In[307]:


print("Accuracy: ", accuracy_score(Y_test_SMOTE,predictions))

#print("F1 score: ", f1_score(y_test,predictions,pos_label='positive',average='micro'))
#print("Recall: ", recall_score(y_test,predictions,pos_label='positive',average='micro'))
#print("Precision: ", precision_score(y_test,predictions,pos_label='positive',average='micro'))

print("\n")
print(classification_report(Y_test_SMOTE,predictions))
print("\n")
print(confusion_matrix(Y_test_SMOTE,predictions))


# In[ ]:




