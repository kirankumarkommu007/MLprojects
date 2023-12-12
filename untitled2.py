import pandas as pd
df = pd.read_csv('Diabetes_dataset.csv')
y = df['Outcome']
X = df.drop('Outcome',axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.70)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cnfmtx = confusion_matrix(y_test,y_pred)
print(cnfmtx)
acc = accuracy_score(y_test,y_pred)
print(acc)
clf = classification_report(y_test,y_pred)
print(clf)



