#Breast Cancer Classification

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer= load_breast_cancer()
cancer.keys()
#Converting the dataset to dataframe
dataset_df= pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target'])) 

#Visualizing the dataset
sns.countplot(dataset_df['target']) 

plt.figure(figsize=(20,10))
sns.heatmap(dataset_df.corr(), annot= True)

#Creating the variables
X= dataset_df.iloc[:, :-1].values
y= dataset_df.iloc[:, 30].values

#Splitting the data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)


#Fitting the model to training set
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
classifier= SVC()
classifier.fit(X_train,Y_train)

y_pred= classifier.predict(X_test)
cm= confusion_matrix(y_test, y_pred)

#Printing the Classification report
print(classification_report(y_test, y_pred))


