# mean average precision score

from collections import defaultdict
import numpy
import sys
import os
import subprocess
import csv
network = sys.argv[1]

alpha1 = int(sys.argv[2])
alpha2 = int(sys.argv[3])

beta1 = int(sys.argv[4])
beta2 = int(sys.argv[5])

gamma1 = int(sys.argv[6])
gamma2 = int(sys.argv[7])
gamma3 = int(sys.argv[8])


scores = defaultdict(list)
fnames = os.listdir("../results/")
print(fnames)
for fname in fnames:
    if fname not in network:
        continue
    if "result" in fname:
        continue
    f = open("../results/%s/%s-fng-sorted-users-%s-%s-%s-%s-%s-%s-%s.csv" % (fname , fname,alpha1,alpha2,beta1,beta2,gamma1,gamma2,gamma3), "r")
    # f = open("../results/amazon/amazon-fng-sorted-users-2-0-2-2-2-2-2.csv", "r")
    for l in f:
        # print(l)
        l = l.strip().split(",")
        if l[1] == "nan":
            l[1] = "0"
        scores[l[0]].append(float(l[1]))
        if l[2] == "nan":
            l[2] = "0"
        scores[l[0]].append(float(l[2]))


f = open("../data/%s_gt.csv" % network,"r")
data = csv.reader(f)

X = []
Y = []

for l in f:
    l = l.strip().split(",")
    d = scores[ l[0]]

    if d == []:
        continue
    if l[1] == "-1":
        Y.append(1)
        X.append(scores[l[0]])
    else:
        Y.append(0)
        X.append(scores[l[0]])
f.close()

# train random forest classifier 
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

X = numpy.array(X)
Y = numpy.array(Y)

X, Y = shuffle(X, Y)
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, Y)

scores = []
aucscores = []
for train, test in skf.split(X,Y):
    train_X = X[train]
    train_Y = Y[train]
    test_X = X[test]
    test_Y = Y[test]

    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(train_X, train_Y)
    scores.append(accuracy_score(test_Y, clf.predict(test_X)))
    try:
        pred_Y = clf.predict_proba(test_X)
        false_positive_rate, true_positive_rate, th =  roc_curve(test_Y, pred_Y[:,1])
        aucscores.append(auc(false_positive_rate, true_positive_rate))
    except:
        pass
    print (scores[-1], aucscores[-1])

print ("Accuracy scores", scores, numpy.mean(scores))
print ("AUC scores", aucscores, numpy.mean(aucscores))
