import coremltools

keras_model_path = 'train.hdf5'

def convert_keras_to_mlmodel(keras_url, mlmodel_url):
    """This method simply converts the keras model to a mlmodel using coremltools.
    keras_url - The URL the keras model will be loaded.
    mlmodel_url - the URL the Core ML model will be saved.
    """
    from keras.models import load_model
    keras_model = load_model(keras_url)

    from coremltools.converters import keras as keras_converter
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    mlmodel = keras_converter.convert(keras_model, input_names=['image'],
                                      output_names=['digitProbabilities'],
                                      class_labels=class_labels,
                                      predicted_feature_name='digit')

    mlmodel.save(mlmodel_url)


coreml_model_path = 'lcd.mlmodel'
convert_keras_to_mlmodel(keras_model_path, coreml_model_path)

spec = coremltools.utils.load_spec(coreml_model_path)
builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)
neuralnetwork_spec = builder.spec

# change the input so the model can accept 28x28 grayscale images
neuralnetwork_spec.description.input[0].type.imageType.width = 64
neuralnetwork_spec.description.input[0].type.imageType.height = 64

from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
grayscale = _FeatureTypes_pb2.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
neuralnetwork_spec.description.input[0].type.imageType.colorSpace = grayscale

builder.inspect_input_features()

from coremltools.models import MLModel

mlmodel_updatable_path = 'LcdModel.mlmodel'

mlmodel_updatable = MLModel(neuralnetwork_spec)
mlmodel_updatable.save(mlmodel_updatable_path)

#coremltools.converters.keras.convert('train.hdf5', class_labels='labels.txt').save('lcd.mlmodel')
