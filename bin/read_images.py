import sys
from keras.preprocessing.image import array_to_img,img_to_array,load_img
from keras.layers import Activation, Dense, Flatten
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
        tmp_img=load_img(d+"/"+f, target_size=(64,64))
        tmp_img_array=img_to_array(tmp_img)
        tmp_img_array/=255
        X_train.append(tmp_img_array)
        Y_train.append(int(dp[2]))
        if f in test_files:
            X_test.append(tmp_img_array)
            Y_test.append(int(dp[2]))
            m=m+1
        n=n+1

X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
X_train=X_train.reshape(112, 12288)
X_test=X_test.reshape(12, 12288)

np.savez("./led-dataset.npz",x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test)

categories=["0","1","2","3","4","5","6","7","8","9"]
nb_classes=len(categories)

Y_train=np_utils.to_categorical(Y_train,nb_classes)
Y_test=np_utils.to_categorical(Y_test,nb_classes)

model=models.Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,Y_train,
          batch_size=100,
          epochs=12,
          verbose=1)

score=model.evaluate(X_test,Y_test)
print(score)
