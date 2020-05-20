import pandas as pd

from pandas import read_csv

import numpy as np
from support import *
import time as t

train = read_csv('dataset/database.csv')
test = read_csv('dataset/testData.csv')

train = train.iloc[:,1:3]
test = test.iloc[:,1:3]


database = {}
testData = {}

for i in range(len(train)):
	k = train.iloc[i,1].split()
	k = k[1:]
	x = k[-1]
	x = x[:-2]
	k[-1] = float(x)
	database[train.iloc[i,0]] = np.array(k,dtype='float64')
	print(k[-1])

for i in range(len(test)):
	k = test.iloc[i,1].split()
	k = k[1:]
	if k[-1]==']]':
		k = k[:-1]
	else:
		x = k[-1]
		x = x[:-2]
		k[-1] = float(x)
	testData[test.iloc[i,0]] = np.array(k,dtype='float64')
	print(k[-1])



def writer(n):
	str_py = n.upper()
	ff = 3	
	pensize_var = 4	
	print('Maximise The New Window')

	name(pensize_var,ff)
	l=0
	for i in range(len(str_py)):
		if (i%(11*ff)==0 and i>0):
			l=l+1
		tab(120*(i%(11*ff))/ff-700,200+120-120/ff-(l*200)/ff)
		if i==0:
			t.sleep(1)
		if str_py[i]==' ':
			continue
		eval(str_py[i]+'()')   	
	t.sleep(2)



def who_is_it(it, database,testData):    
    encoding = testData[it]
    #print(encoding)
    min_dist = 100
    for (name, db_enc) in database.items():
        #print(len(db_enc))
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    
    
    if min_dist > 0.7:
        writer("not in database")
    else:
        writer(str(identity))
        
    return min_dist, identity




for i in range(15):
	print("Show me your face : ")
	it = str(input())
	image = 'dataset/test/'+ it
	distance , name = who_is_it(it,database,testData)
	writer(str(name))
	print("\nName in database "+str(name)+" distance "+str(distance)+"\n")


