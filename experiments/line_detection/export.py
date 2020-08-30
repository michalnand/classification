import sys
sys.path.insert(0,'../..')

import libs

model_path = "./models/net_0/"
import models.net_0.model as Net0

input_shape     = (1, 512, 512)
outputs_count   = 1

model = Net0.Model(input_shape, outputs_count)
model.load(model_path + "trained/")


export = libs.ExportNetwork(model, input_shape, export_path = model_path + "/export/", network_prefix = "LineNetwork")



