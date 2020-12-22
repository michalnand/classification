import sys
sys.path.insert(0,'../..')

import embedded_inference.libs_embedded


input_shape     = (4, 512)
output_shape    = (5, )

'''
supported bits : 
-1  : float
8   : int8_t
16  : int16_t
32  : int32_t

supported quantization mode : 
1, "sigma_1", covering 65% of weights
2, "sigma_2", covering 95% of weights
3, "sigma_3", covering 99.7% of weights
4, "all",     covering 100% of weights
'''



'''
model_path = "./models/net_0/"
import models.net_0.model as Net0

model = Net0.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=32, quantization_mode="sigma_2")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")

model_path = "./models/net_1/"
import models.net_1.model as Net1

model = Net1.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=32, quantization_mode="sigma_2")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")




model_path = "./models/net_2/"
import models.net_2.model as Net2

model = Net2.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=32, quantization_mode="sigma_2")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")





model_path = "./models/net_3/"
import models.net_3.model as Net3

model = Net3.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=32, quantization_mode="sigma_2")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")




model_path = "./models/net_4/"
import models.net_4.model as Net4

model = Net4.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=32, quantization_mode="sigma_2")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")


model_path = "./models/net_5/"
import models.net_5.model as Net5

model = Net5.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=32, quantization_mode="sigma_2")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")

'''



model_path = "./models/net_4/"
import models.net_4.model as Net4

model = Net4.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=8, quantization_mode="sigma_2")
 

'''
model_path = "./models/net_6/"
import models.net_6.model as Net6

model = Net6.Create(input_shape, output_shape)
model.load(model_path + "trained/")

export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkFloat", io_bits=-1, weights_bits=-1, accumulation_bits=-1, quantization_mode="all")
export = embedded_inference.libs_embedded.ExportModel(model, input_shape, export_path = model_path + "/export/", network_prefix = "MagnetometetNetworkInt8", io_bits=8, weights_bits=8, accumulation_bits=32, quantization_mode="sigma_2")
'''