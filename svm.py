#coding=utf-8
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import pickle
import gzip
import numpy as np
import time
f =gzip.open('./DetectionBinsData_pickle615_clean.gzip','rb') 
clf = SGDClassifier(max_iter=10000)
#clf = svm.SVC()

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
clf.fit(X, Y)  

start = time.time()
acc_set=[]
for p in range(10):
    #test step
    error_d=0
    error_b=0
    for i in range(100):
        test=pickle.load(f)
        d=test[:,1:101]
        b=test[:,102:]  
        result_d = clf.predict(d) # 0 for dark; 1 for bright
        result_b = clf.predict(b) # 0 for dark; 1 for bright
        #print(result_d, '\n', result_b)
        
        for j in range(100):
            if result_d[j]!=0:
                error_d+=1
            if result_b[j]!=1:
                error_b+=1
    #print('dark error counts:', error_d)
    #print('bright error counts:', error_b)
    accuracy_rate=1-(float)(error_b+error_d)/20000.0
    #print('total accuracy rate:',accuracy_rate)   #98.56%
    acc_set.append(accuracy_rate)
print(float(np.mean(acc_set)),float(np.mean(acc_set)-np.min(acc_set)),float(np.max(acc_set)-np.mean(acc_set))) 
f.close()

end = time.time()
print ('Time used: ',end-start)