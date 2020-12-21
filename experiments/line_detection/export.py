import sys
sys.path.insert(0,'../..')

import embedded_inference.libs_embedded

model_path = "./models/net_0/"
import models.net_0.model as Net0

input_shape     = (1, 512, 512)
outputs_count   = 1

model = Net0.Model(input_shape, outputs_count)
model.load(model_path + "trained/")


export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "LineNetwork", io_bits = 8, weights_bits = 8, accumulation_bits = 32, quantization_mode = "sigma_2")



