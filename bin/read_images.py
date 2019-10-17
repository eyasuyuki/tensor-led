from keras.preprocessing.image import img_to_array,load_img
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras import models
from keras.utils import np_utils
import numpy as np
import os

X_train = []
Y_train = []

X_test = []
Y_test = []

dirs = ["./images/0",
      "./images/1",
      "./images/2",
      "./images/3",
      "./images/4",
      "./images/5",
      "./images/6",
      "./images/7",
      "./images/8",
      "./images/9"]

test_files=["n0.png",
            "n1.png",
            "n2.png",
            "n3.png",
            "n4.png",
            "n5.png",
            "n6.png",
            "n6b.png",
            "n7.png",
            "n7b.png",
            "n8.png",
            "n9.png"]

n = 0
m = 0
for d in dirs:
    dp = d.split("/")
    fs = os.listdir(d)
    for f in fs:
        tmp_img = load_img(d+"/"+f, grayscale=True, target_size=(64, 64))
        tmp_img_array = img_to_array(tmp_img)
        tmp_img_array /= 255
        print(f"{tmp_img_array.shape}")
        X_train.append(tmp_img_array)
        Y_train.append(int(dp[2]))
        if f in test_files:
            X_test.append(tmp_img_array)
            Y_test.append(int(dp[2]))
            m = m+1
        n = n+1

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

np.savez("./led-dataset.npz", x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)

categories=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nb_classes = len(categories)

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = models.Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 1)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

json_text = model.to_json()
print(json_text)
open("train.json", "w").write(json_text)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=100,
          epochs=12,
          verbose=1)

model.save_weights("train.hdf5")

score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
