#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as np
import pandas as pd
import math
import random


# In[2]:


listOfHeaders = ['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315_of_diluted wines','Proline']
dataset = pd.read_csv('wine.data',names=listOfHeaders)


# In[3]:


print(dataset.shape)
dataset.isnull().sum()


# In[12]:


dataset


# In[7]:


from sklearn.model_selection import train_test_split

inputs = dataset[['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids',
           'Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315_of_diluted wines']]

target = dataset['Proline']
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.3)


# In[14]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
model = clf.fit(X_train, y_train)

resultDT = model.predict(X_test) 

from sklearn import metrics

print(resultDT)
print(y_test)


# In[15]:


# inputs
X = dataset
N_List = listOfHeaders
NAnt = 10
M = 5
NCmax = 50
NF = 6
p = 0.2
c = 0.2
q0 = 0.7
# B = 1


# In[16]:


# extra mini methods

# similarity : 
def sim(A,B):
    sumInterMult = 0
    for i in range(0,len(A)):
        sumInterMult+= (A[i] * B[i])
    sqrtSumPowA = 0
    for i in range(0,len(A)):
        sqrtSumPowA+= A[i]**2
    sqrtSumPowB = 0
    for i in range(0,len(B)):
        sqrtSumPowB+= B[i]**2
    res = abs((sumInterMult)/((math.sqrt(sqrtSumPowA))*(math.sqrt(sqrtSumPowB))))
    return res

# choose randomly 
def RndomChoose(jki,k,X,T,N_list):
    thisPos = jki[k]
    jki.pop(k)
    temp = []
    sumTemps = 0
    for e in jki:
        temp_ = T[(len(T)-1)][e]/sim(list(X[N_list[thisPos]]),list(X[N_list[e]]))
        temp.append(temp_)
        sumTemps += temp_
    Pk_any = []
    sumP = 0
    for e in temp:
        Pk_any.append(e/sumTemps)
        sumP+=(e/sumTemps)
    randP = random.random()*sumP
    ii=0
    for e in Pk_any:
        if randP-e <=0:
            return jki[ii]
        else:
            randP -= e
        ii+=1
    return -1
# choose randomly 
def GreedyChoose(jki,k,X,T,N_list):
    thisPos = jki[k]
    jki.pop(k)
    maxRes = 0
    res = -1
    for e in jki:
        temp_ = T[(len(T)-1)][e]/sim(list(X[N_list[thisPos]]),list(X[N_list[e]]))
        if maxRes < temp_:
            res = e
    return res

def sortSecond(val): 
    return val[1] 


# In[ ]:





# In[17]:


# main method
def UFSACO(X,N_list,NAnt,M,NCmax,NF,p,c,q0):
    n = len(N_list)

    # matrix of similarity
    similarities = [[0]*n]*n
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            similarities[i][j] = sim(list(X[N_list[i]]),list(X[N_list[j]]))
    
    # set pheremones
    T = [[c]*n]*1
    
    # iterate loop
    for t in range(NCmax):
        FC = [0]*n
        
        # select m features randomly
        listOfNodes = random.sample(N_list, NAnt)
        for i in range(NF):
            for k in range(NAnt):
                # find q randomly
                q = random.random()
                Jki = list(range(len(listOfNodes)))
                # condition 
                if q > q0:
                    # define method to choose randomly
                    destination = RndomChoose(Jki,k,X,T,N_list)
                else:
                    # define method to choose greedy
                    destination = GreedyChoose(Jki,k,X,T,N_list)
                destination1 = N_list.index(listOfNodes[destination])
                FC[destination1] += 1
        tempList = []
        ii = 0
        for e in T[t]:
            tempList.append(((1-p)*e)+(FC[ii]/sum(FC)))
            ii+=1
        T.append(tempList)
    ii2 = 0
    resultList = []
    for el in T[len(T)-1]:
        resultList.append([ii2,el])
        ii2+=1
    print(resultList)
    resultList.sort(key = sortSecond,reverse=True)
    return resultList[:M]


# In[18]:


res = UFSACO(X,N_List,NAnt,M,NCmax,NF,p,c,q0)


# In[19]:


res


# In[20]:


res2 = []
for el in res:
    res2.append(listOfHeaders[el[0]])


# In[21]:


res2


# In[22]:


df = dataset[res2]


# In[23]:


df


# In[24]:


from sklearn.model_selection import train_test_split

inputs = dataset[['Alcalinity_of_ash', 'Malic_acid', 'Color_intensity', 'Proanthocyanins', 'Total_phenols']]

target = dataset['Proline']
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.3)


# In[26]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
model = clf.fit(X_train, y_train)

resultDT = model.predict(X_test) 

from sklearn import metrics

print(resultDT)
print(y_test)


# In[ ]:





# In[ ]:




