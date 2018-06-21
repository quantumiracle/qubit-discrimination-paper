#coding=utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import pickle
import gzip
import numpy as np
f =gzip.open('./DetectionBinsData_pickle61_clean.gzip','rb') 

'''classification model'''
#clf = SGDClassifier(max_iter=10000)
#clf = svm.SVC()
#model = LogisticRegression()   # %98.5%
#model = GaussianNB()   # 95.7%
#model = KNeighborsClassifier()  #%92.37
#model = DecisionTreeClassifier()  #%97.33

#learn step
X=[]
Y=[]
batch_num=600
for i in range (batch_num):   #training times
    print(i)
    data=pickle.load(f)
    d=data[:,1:101]
    b=data[:,102:]
    batch=[]
    batch.append(d)
    batch.append(b)
    #print(batch)
    train_batch=np.array(batch).reshape(200,100)
    #print(train_batch) 
    train_Y=100*[0]+100*[1]
    #print(Y)
    X.append(train_batch)
    Y.append(train_Y)
    
    
X=np.array(X).reshape(batch_num*200,100)
Y=np.array(Y).reshape(batch_num*200)
#http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
model.fit(X, Y)  

#test step
error_d=0
error_b=0
for i in range(100):
    test=pickle.load(f)
    d=test[:,1:101]
    b=test[:,102:]  
    result_d = model.predict(d) # 0 for dark; 1 for bright
    result_b = model.predict(b) # 0 for dark; 1 for bright
    #print(result_d, '\n', result_b)
    
    for j in range(100):
        if result_d[j]!=0:
            error_d+=1
        if result_b[j]!=1:
            error_b+=1
print('dark error counts:', error_d)
print('bright error counts:', error_b)
accuracy_rate=1-(float)(error_b+error_d)/20000.0
print('total accuracy rate:',accuracy_rate)  
f.close()