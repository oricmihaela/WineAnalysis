
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
sns.set_theme(style="darkgrid")




#load red wine dataset
wine_red = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep = ';')
#add column to indicate red wine
wine_red['wine_type'] = 0

#load white wine dataset
wine_white = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep = ';')
#add column to indicate red wine
wine_white['wine_type'] = 1


#Concatenate datasets
wine = pd.concat([wine_red,wine_white], axis=0)

#print first 5 rows
print('*****First 5 rows*****')
print(wine.head())

#print no of instances
print('*****Wine quality count*****')
print(wine['quality'].value_counts())

print('*****Wine type count*****')
print(wine['wine_type'].value_counts())

#print stats
print('*****Statistics*****')
print(wine.describe())









#TO DO: Read about seaborn and display statistical results/distributions etc  (http://seaborn.pydata.org/)
#fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide    density           pH    sulphates      alcohol      quality    wine_type

wine.hist(bins=25,figsize=(10,10))
sns.histplot(x="quality", y="wine_type", data=wine)
sns.lmplot(x="quality", y="alcohol", hue="wine_type", data=wine)
sns.lmplot(x="pH", y="chlorides", hue="wine_type", data=wine)
sns.lmplot(x="quality", y="alcohol", hue="wine_type", data=wine)



#Creating the datasets as numpy arrays
X = np.array(wine.drop(labels={'quality','wine_type'}, axis=1))
y_quality = np.array(wine['quality'])
y_type = np.array(wine['wine_type'])



# print('****************** WINE TYPES ******************')

#Classify according to wine type
y = y_type

#Divide data into training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#Standardize dataset (ie mean=0 and std=1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Compare random forest to support vector machine

#Random forest classifier
print('Random forest classifier')
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
rfc_train = rfc.predict(X_train)
rfc_pred = rfc.predict(X_test)


print('Train set')
print(classification_report(y_train, rfc_train))
print(confusion_matrix(y_train, rfc_train))

print('Test set')
print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))



# SVM Classifier
print('SVM classifier')
svm = SVC(random_state=32)
svm.fit(X_train, y_train)
svm_train = rfc.predict(X_train)
svm_pred = rfc.predict(X_test)


print('Train set')
print(classification_report(y_train, svm_train))
print(confusion_matrix(y_train, svm_train))

print('Test set')
print(classification_report(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))


print('****************** WINE QUALITY ******************')

#Classify according to wine quality
y = y_quality

#Convert wine quality to three levels (low, medium, high)
#low 0-5 -> 0
y[y<=5]=0
#medium 6-7 -> 1
y[(y>=6) & (y<=7)]=1
#high 8-10 -> 2
y[y>=8]=2

#Divide data into training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#Standardize dataset (ie mean=0 and std=1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Compare random forest to support vector machine

#Since we have more than two labels we shall use one-vs-rest OR one-vs-all approach
#Read more about this:
#https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/
#https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html


#Random forest classifier
print('Random forest classifier')
#initialize rf
rfc = RandomForestClassifier(n_estimators = 100)
#initialize ovr strategy
ovr_rfc = OneVsRestClassifier(rfc).fit(X_train, y_train)
ovr_rfc.fit(X_train, y_train)
ovr_rfc_train = ovr_rfc.predict(X_train)
ovr_rfc_pred = ovr_rfc.predict(X_test)


print('Train set')
print(classification_report(y_train, ovr_rfc_train))
print(confusion_matrix(y_train, ovr_rfc_train))

print('Test set')
print(classification_report(y_test, ovr_rfc_pred))
print(confusion_matrix(y_test, ovr_rfc_pred))



# SVM Classifier
print('SVM classifier')
#initialize svm
svm = SVC(random_state=32)
#initialize ovr strategy
ovr_svm = OneVsRestClassifier(rfc).fit(X_train, y_train)
ovr_svm.fit(X_train, y_train)
ovr_svm_train = ovr_svm.predict(X_train)
ovr_svm_pred = ovr_svm.predict(X_test)


print('Train set')
print(classification_report(y_train, ovr_svm_train))
print(confusion_matrix(y_train, ovr_svm_train))

print('Test set')
print(classification_report(y_test, ovr_svm_pred))
print(confusion_matrix(y_test, ovr_svm_pred))


plt.show()


