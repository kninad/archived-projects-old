from os import listdir
import sys
import re
from sets import Set
import math 

def process(folder):
  users = dict()
  with open(folder,"r") as f:
    for line in f:
      a = line.strip("\r\n").split(",") 
      if not users.has_key(a[1]): 
        users[int(a[1])] = dict()
      users[int(a[1])][int(a[0])] = float(a[2])
  return users
        
def calMean():
  global users
  means = dict()
  for x in users:
    total = 0.0
    for d in users[x]:
      total = total + users[x][d]
    means[x] = total/len(users[x])
  return means
  
def calWeights():
  w = dict()
  global users
  global vbar
  for a in users:
    w[a] = dict()
    for i in users:
      num = 0.0
      den1 = 0.0
      den2 = 0.0
      if not a == i and not(w.has_key(a) and w[a].has_key(i)):
        for j in users[a]:
          if users[i].has_key(j):
            one= users[a][j] - vbar[a]
            two= users[i][j] - vbar[i]
            thr= one*one
            four= two*two
            num = num + (one*two)
            den1 = den1 + thr
            den2 = den2 + four
        if den1 == 0 or den2 == 0: w[a][i] = 0.0
        else:
          w[a][i] = num / math.sqrt(den1*den2)
        
        if(not w.has_key(i)): 
          w[i] = dict()
        w[i][a] = w[a][i]
  return w

def predict(K):
  global testset
  global W
  res = dict()
  for a in testset:
    res[a] = dict()
    for j in testset[a]:
      total = 0.0
      predicted = 0.0
      if W.has_key(a):
        for i in W[a]:
          if users[i].has_key(j): 
            Vij = users[i][j]
            total = total + (W[a][i] * (Vij - vbar[i])) 
        predicted = vbar[a] + (K * total)
        res[a][j] = predicted 
  return res

def meanAbsEr(set1, set2):
  er = 0.0
  size = 0.0
  for a in set2:
    for b in set2[a]:
      er = er + abs(set1[a][b] - set2[a][b])
      size = size + 1  
  return er/size

def meanSqdEr(set1, set2):
  er = 0.0
  size = 0.0
  for a in set2:
    for b in set2[a]:
      er = er + (set1[a][b] - set2[a][b])**2
      size = size + 1
  er = er/size  
  return math.sqrt(er)

users = process(sys.argv[1])
vbar = calMean()
W = calWeights()
kappa = sys.argv[3]
testset = process(sys.argv[2])
predicted = predict(kappa)
print ("MAE: ", meanAbsEr(testset,predicted))
print ("RMSE: ",meanSqdEr(testset,predicted))
