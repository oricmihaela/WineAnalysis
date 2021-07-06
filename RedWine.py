#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


#loading dataset
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep = ';')


# In[3]:


wine.head()


# In[4]:


wine['quality'].value_counts()


# In[9]:


#dijelimo vina na 3 kategorije, bins = ((prva granica), ..., (zadnja granica))
#broj labela mora bit za jedan manji od broja parametara kod dijeljenja binsa
bins = (3, 4.5, 6.5, 8)
group_names = ['bad', 'average', 'great']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[10]:


wine['quality'].unique()


# In[13]:


#onaj encoder sto smo importali da mozemo losa vina kategorizirat kao 0, prosjecna kao 1, odlicna kao 2
label_quality = LabelEncoder()


# In[16]:


#primjena encodera na dataset
#prvo iskoristimo funkciju fit transform da promijenimo kvalitete u one zadane u encoderu(0, 1, 2) i onda ih ubacujemo na mjesto wine['quality']
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[17]:


wine.head()


# In[18]:


wine['quality'].unique()


# In[19]:


wine['quality'].value_counts()


# In[20]:


#ukupno imamo 1599 vina, ali nakon podjele njih je 1589 sto znaci da neke vrijednosti nisu uvrstene u podjelu


# In[56]:


wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep = ';')


# In[57]:


wine['quality'].value_counts()


# In[58]:


bins = (0, 4, 6, 9)
group_names = ['bad', 'average', 'great']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[59]:


# In[60]:


#provjera je li dobro podijeljeno tako da zbrojimo 3 i 4, 5 i 6, 7 i 8
#3+4=63, 5+6=1319, 7+8=217
#dobra podjela


# In[61]:


label_quality = LabelEncoder()


# In[62]:


wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[63]:



# In[64]:


#vidimo da encoder ne radi dobro jer vina srednje kvalitete oznacava kao 0


# In[65]:


#iz prvih 200 rezutata moze se zakljuciti da su losa vina oznacena kao 1, a prosjcena kao 0
wine.head(200)


# In[67]:


#label encoder koristi abecedni poredak da bi odredio sto je 0, 1, 2
#ako zelimo da nam je bad = 0, average = 1, great = 2, moramo im promijeniti nazive


# In[68]:


wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep = ';')


# In[69]:


bins = (0, 4, 6, 9)
group_names = ['0bad', '1average', '2great']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[70]:


wine['quality'].value_counts()


# In[71]:


wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[72]:



# In[73]:


#uspjecno encodirano


# In[74]:


#razdvajanje podataka na ulazne (X) i izlazne(y) varijable
X = wine.drop('quality', axis=1)
#axis 1 znaci da radimo sa stupcima, samo izbacujemo stupac s kvalitetama
y = wine['quality']
#samo kvaliteta vina


# In[75]:


#razdvajanje trening i test seta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)


# In[76]:


#Neki ulazi su u puno većem rasponu od drugih pa da ti manji ulazi ne bi bili beznačajni, sve se skalira na vrijednosti od 0 do 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[77]:


X_train[:5]


# In[78]:


#provjera jel imamo podjednak omjer kvaliteta u trening i testing setu
y_train.value_counts()


# In[82]:


# 0 => 56/1359 = 4.12%
# 1 => 1123/1359 = 82.63%
# 2 => 180/1359 = 13.24%


# In[80]:


y_test.value_counts()


# In[81]:


# 0 => 7/240 = 2.92%
# 1 => 196/240 = 81.67%
# 2 => 37/240 = 15.41%


# Random Forest Classifier

# In[106]:


rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[107]:


print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# SVM Classifier

# In[114]:


clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)


# In[115]:


print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))


# Neural Network

# In[116]:


mlpc = MLPClassifier(hidden_layer_sizes = (15, 3), max_iter = 2000) 
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)


# In[117]:


print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# In[119]:


from sklearn.metrics import accuracy_score
cm_rfc = accuracy_score(y_test, pred_rfc)
cm_rfc


# In[120]:


from sklearn.metrics import accuracy_score
cm_clf = accuracy_score(y_test, pred_clf)
cm_clf


# In[121]:


from sklearn.metrics import accuracy_score
cm_mlpc = accuracy_score(y_test, pred_mlpc)
cm_mlpc


# In[ ]:




