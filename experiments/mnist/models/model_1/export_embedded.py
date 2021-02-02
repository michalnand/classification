import sys
sys.path.insert(0,'../../../..')

import embedded_inference.libs_embedded


input_shape     = (1, 28, 28)
output_shape    = (10, )



import model

model_path      = "./"
pretrained_path = model_path + "trained/"
export_path     = model_path + "export/"


embedded_inference.libs_embedded.ExportModel("MyModel", input_shape, output_shape, model, pretrained_path, export_path, False)
