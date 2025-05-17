import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
tn=pd.read_csv('titanic_train.csv')
#print(tn.head(10))
#print(tn.isnull().head(10))

#sns.heatmap(tn.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
#sns.countplot(x='Survived',data=tn,palette='rainbow')
#sns.countplot(x='Survived',hue='sex',data=tn,palette='rainbow')
#sns.countplot(x='Survived',hue='Pclass',data=tn,palette='rainbow')
#sns.displot(tn['Age'].dropna(),kde=False,color='darkred',bins=40)
#sns.countplot(x='SibSp',data=tn)
#tn['Fare'].hist(color='green',bins=40,figsize=(8,4))
#pt.figure(figsize=(12,7))
#sns.boxplot(x='Pclass',y='Age',data=tn,palette='winter')

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


tn.drop('Cabin',axis=1,inplace=True)
print(tn.head(5))
tn['Age']=tn[['Age','Pclass']].apply(impute_age,axis=1)
#sns.heatmap(tn.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print(pd.get_dummies(tn['Embarked'],drop_first=True).head())
sex=pd.get_dummies(tn['Sex'],drop_first=True).astype(float)
embark=pd.get_dummies(tn['Embarked'],drop_first=True).astype(float)
tn.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
print(tn.head(5))
tn=pd.concat([tn,sex,embark],axis=1)
print(tn.head(5))
X_train,X_test,Y_train,Y_test=train_test_split(tn.drop('Survived',axis=1),tn['Survived'],test_size=0.3,random_state=101)
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)
pred=logmodel.predict(X_test)

accuracy=confusion_matrix(Y_test,pred)
print(accuracy)
accracy=accuracy_score(Y_test,pred)
print(accracy)
print(pred)
pt.show()