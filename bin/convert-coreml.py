import coremltools

coremltools.converters.keras.convert('train.hdf5', class_labels='labels.txt').save('lcd.mlmodel')
