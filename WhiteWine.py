#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep = ';')


# In[3]:


wine.head()


# In[4]:


wine['quality'].value_counts()


# In[5]:


bins = (0, 4, 6, 9)
group_names = ['0bad', '1average', '2great']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[6]:


wine['quality'].value_counts()


# In[7]:


label_quality = LabelEncoder()


# In[8]:


wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[11]:


wine.head(20)


# In[12]:


#razdvajanje podataka na ulazne (X) i izlazne(y) varijable
X = wine.drop('quality', axis=1)
#axis 1 znaci da radimo sa stupcima, samo izbacujemo stupac s kvalitetama
y = wine['quality']
#samo kvaliteta vina


# In[13]:


#razdvajanje trening i test seta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)


# In[14]:


#Neki ulazi su u puno većem rasponu od drugih pa da ti manji ulazi ne bi bili beznačajni, sve se skalira na vrijednosti od 0 do 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


y_train.value_counts()


# In[16]:


# 0 => 158/4163 = 3,79%
# 1 => 3119/4163 = 74.92%
# 2 => 886/4163 = 21.28%


# In[17]:


y_test.value_counts()


# In[18]:


# 0 => 25/735 = 3.4%
# 1 => 536/735 = 72.93%
# 2 => 174/735 = 23.67%


# Random Forest Classifier

# In[19]:


rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[20]:


print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# SVM Classifier

# In[21]:


clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)


# In[22]:


print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))


# Neural Network

# In[23]:


mlpc = MLPClassifier(hidden_layer_sizes = (15, 3), max_iter = 2000) 
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)


# In[24]:


print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# In[25]:


from sklearn.metrics import accuracy_score
cm_rfc = accuracy_score(y_test, pred_rfc)
cm_rfc


# In[26]:


from sklearn.metrics import accuracy_score
cm_clf = accuracy_score(y_test, pred_clf)
cm_clf


# In[27]:


from sklearn.metrics import accuracy_score
cm_mlpc = accuracy_score(y_test, pred_mlpc)
cm_mlpc


# In[ ]:




