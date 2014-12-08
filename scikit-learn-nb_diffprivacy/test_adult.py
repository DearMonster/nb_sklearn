#! /usr/bin/python

__author__ = 'deng'

'''
  run the dataset adult with CART algorithm.
'''

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from datetime import datetime
from sklearn import cross_validation
from sklearn import tree


adult = pd.read_table("adult.all.data", sep=',',header=None,na_values=' ?',
                      names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                             'hours-per-week', 'native-country', 'class'])
adult = adult.dropna()
print('adult length: %d' % len(adult))

cols = [c for c in adult.columns if c != 'class']
x = adult.ix[:, cols]
y = adult.ix[:, 'class']


v = DictVectorizer()

xx = v.fit_transform(x.to_dict(orient='records')).toarray()


le = preprocessing.LabelEncoder()
yy = le.fit_transform(y)


data_train, data_test, target_train, target_test = train_test_split(xx, yy)

print('algorithm begin')


dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)



# for i in range(2, 11):
start_time = datetime.now()
dtc = dtc.fit(xx,yy)
#scores = cross_validation.cross_val_score(dtc, xx, yy, cv=10)  # 10-fold cv
scores = cross_validation.cross_val_score(dtc, xx, yy, cv=10)
print("%s Cross Avg. Score: %0.2f (+/- %0.2f)" % (10, scores.mean(), scores.std() * 2))
end_time = datetime.now()
time_spend = end_time - start_time
print("%d Time: %0.2f" % (10, time_spend.total_seconds()))
print(scores)
print(np.sum(scores) / 10)
print("xxx")
