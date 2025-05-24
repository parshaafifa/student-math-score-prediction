import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tools.tools as stattools
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,cohen_kappa_score,classification_report,mean_squared_error ,confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from  sklearn import linear_model
Df=pd.read_csv("C:/Users/User/Downloads/Kaggle dataset/StudentsPerformance.csv")
D=pd.get_dummies(Df,drop_first=True)
print(D.dtypes)
Y=D['math score']
X=D.drop(['math score'],axis=1)
model=linear_model.LinearRegression()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
Model=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
r2_score(Y_test,Y_pred)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted Math Scores")
plt.show()


