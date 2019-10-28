import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("./train.hdf5")
tflite_model = converter.convert()
open("train.tflite", "wb").write(tflite_model)

