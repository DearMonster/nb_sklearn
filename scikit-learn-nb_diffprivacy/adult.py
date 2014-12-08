__author__ = 'same'

import numpy as np  
import scipy as sp  
from sklearn import tree  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.cross_validation import train_test_split  , cross_val_score
  
   
# data read
data   = []  
labels = []  
with open("/home/same/python/adult.txt") as ifile:
        nline=0
        for line in ifile:  
            tokens = line.strip().split(' ')
            temp = []
            s=' '
            for tk in tokens[1:]:
                s=' '
                for t in tk:
                    if(t != ':'):
                        s+=t
                    else:
                        break;
                temp.append(s)
            data.append(temp)
            labels.append(tokens[0])  
            nline+=1
#             if(i==5):
#                 break;
print("Line")
print(nline)
# print("data=============")
# print(data)
x = np.array(data)  
labels = np.array(labels)  
# print("x:===================")
# print(x)
# print("label===============")
# print(labels)

X = [[0 for col in range(123)] for row in range(nline)]
i=0
for row in x:
    for j in row:
#         print("i: %d,j: %s" % (i ,j))
        X[i][int(j)-1]=float(1)
    i+=1
X=np.array(X)

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)  
# print("clf:  ")
# print(clf)  
 
 
clf = clf.fit(X, labels)

# X_train,Y_train = load_svmlight_file("/home/same/python/adult.txt")
# X_train.
# clf = clf.fit(XX,YY)
# socres = cross_val_score(clf,X_train,Y_train,cv=10)
socres = cross_val_score(clf,X,labels,cv=10)


print("-------------------------------------------------------------")
print(socres.mean())

# print(X_train)
# print("=======================================")
# print(Y_train)
  
