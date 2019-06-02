import sys
from keras.preprocessing.image import array_to_img,img_to_array,load_img
from keras import layers,models
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import os

X_train=[]
Y_train=[]

X_test=[]
Y_test=[]

dirs=["./images/0","./images/1","./images/2","./images/3","./images/4","./images/5","./images/6","./images/7","./images/8","./images/9"]

test_files=["n0.png","n1.png","n2.png","n3.png","n4.png","n5.png","n6.png","n6b.png","n7.png","n7b.png","n8.png","n9.png"]

n=0
m=0
for d in dirs:
    dp = d.split("/")
    fs=os.listdir(d)
    for f in fs:
        tmp_img=load_img(d+"/"+f)
        tmp_img_array=img_to_array(tmp_img)
        X_train.append(tmp_img_array)
        Y_train.append(dp[2])
        if f in test_files:
            X_test.append(tmp_img_array)
            Y_test.append(dp[2])
            m=m+1
        n=n+1

#X_train=X_train.reshape(n,4096)
#Y_train=Y_train.reshape(m,4096)
print(len(X_train))
print(len(X_test))
X_train=np.array(X_train).astype(np.float32)/255
X_test=np.array(X_test).astype(np.float32)/255

np.savez("./led-dataset.npz",x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test)

categories=["0","1","2","3","4","5","6","7","8","9"]
nb_classes=len(categories)

#f=np.load("./led-dataset.npz",allow_pickle=True)
#X_train1,Y_train1=f['x_train'],f['y_train']
#X_test1,Y_test1=f['x_test'],f['y_test']

y_train=np_utils.to_categorycal(Y_train,nb_classes)
y_test=np_utils.to_catehgorycal(Y_test,nb_classes)

model=models.Sequential()
model.add(Dense(64, activation='relu', input_dim=4096))
model.add(Dense(10, activation='softmax'))

model.compile(optimize='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,
          batch_size=100,
          epochs=12,
          verbose=1)

score=model.evaluate(x_test,y_test)
print(score[0])
print(score[1])
