import coremltools

model = coremltools.models.MLModel('LcdModel.mlmodel')

model.visualize_spec()