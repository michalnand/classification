import sys
sys.path.insert(0,'../..')

import libs


input_shape     = (4, 512)
output_shape    = (4, )


model_path = "./models/net_0/"
import models.net_0.model as Net0

model = Net0.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = libs.ExportNetwork(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetwork")




model_path = "./models/net_1/"
import models.net_1.model as Net1

model = Net1.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = libs.ExportNetwork(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetwork")




model_path = "./models/net_2/"
import models.net_2.model as Net2

model = Net2.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = libs.ExportNetwork(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetwork")





model_path = "./models/net_3/"
import models.net_3.model as Net3

model = Net3.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = libs.ExportNetwork(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetwork")


