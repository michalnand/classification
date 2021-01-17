import sys
sys.path.insert(0,'../..')

import embedded_inference.libs_embedded

import libs_dataset

dataset_path = "/Users/michal/dataset/car_detection_2/"
#dataset_path = "/home/michal/dataset/car_detection_2/"

folders_list = []

folders_list.append(dataset_path + "/Bytca")
#folders_list.append(dataset_path + "/Kysuce")
#folders_list.append(dataset_path + "/Martin_1")
#folders_list.append(dataset_path + "/Martin_2")
dataset = libs_dataset.DatasetMagnetometer2(folders_list, width = 512, augmentations_count = 50, testing_ratio = 50)

input_shape     = dataset.input_shape
output_shape    = dataset.output_shape

training_x, training_y = dataset.get_testing_batch(512)


input_shape     = (4, 512)
output_shape    = (5, )



import models.net_0.model as Net0

model_path      = "./models/net_0/"
pretrained_path = model_path + "trained/"
export_path     = model_path + "export/"


embedded_inference.libs_embedded.ExportModelCode("MyModel", input_shape, output_shape, Net0, pretrained_path, export_path, training_x, training_y, True)
