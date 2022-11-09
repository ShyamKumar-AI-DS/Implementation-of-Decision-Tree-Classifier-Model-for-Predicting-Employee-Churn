# Implementation of Decision Tree Classifier Model for Predicting Employee Churn
# AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
# EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
# ALGORITHM:
1. Import the required packages.
2. Read the data set.
3. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4. Determine training and test data set.
5. Apply decision tree Classifier and get the values of accuracy and data prediction.
# PROGRAM:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shyam Kumar A 
RegisterNumber: 212221230098
*/
```

```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
# OUTPUT:
![op1](https://user-images.githubusercontent.com/93427182/200743875-29dde148-e431-4016-bc5c-e2ba1abe2482.png)
![op2](https://user-images.githubusercontent.com/93427182/200743903-98432cd0-5bf8-4049-8c61-c3a7abc82cd6.png)
![op3](https://user-images.githubusercontent.com/93427182/200743911-5aef8bcd-05a0-40f0-b318-1d6d81fedb9f.png)

# RESULT:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
