import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix,accuracy_score


df=pd.read_csv("zoo_data.csv",header=None)

y=df[16]

X=df.drop(16, axis=1)


for i in range(len(y)):
    if y[i] <= 4:
        y[i]=0
    else:
        y[i]=1



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100,test_size=0.35)
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)
y_pred=clf_dt.predict(X_test)

cm=confusion_matrix(y_pred,y_test)

FP=cm.sum(axis=0)-np.diag(cm)
FN=cm.sum(axis=1)-np.diag(cm)
TP=np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

TPR=TP/(TP+FN)
TNR=TN/(TN+FP)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
accuracy=(TP+TN)/(TP+TN+FP+FN)
f_score=2*(precision*recall)/(precision+recall)

print("False positive:{}".format(FP))
print("False Negative:{}".format(FN))
print("True positive:{}".format(TP))
print("True Negative:{}".format(TN))
print("TPR:{}\n".format(TPR))
print("TNR:{}\n".format(TNR))
print("Precision:{}\n".format(precision))
print("Recall:{}\n".format(recall))
print("Accuracy:{}\n".format(accuracy))
print("Fscore:{}\n".format(f_score))


plot_confusion_matrix(clf_dt, X_test, y_test,cmap=plt.cm.Blues, display_labels=["0", "1"])


plt.figure(figsize=(15,7.5))
plot_tree(clf_dt,
          filled=True, 
          rounded=True, 
          class_names=["No", "Yes"], 
          feature_names=X.columns
          )