import sys
sys.path.insert(0,'../../../..')

import libs
import libs_dataset

import model                as ModelReference
import model_testing_int8   as ModelTestingInt8



dataset_path = "/Users/michal/dataset/car_detection_2/"
#dataset_path = "/home/michal/dataset/car_detection_2/"

folders_list = []

folders_list.append(dataset_path + "/Bytca")
folders_list.append(dataset_path + "/Kysuce")
folders_list.append(dataset_path + "/Martin_1")
folders_list.append(dataset_path + "/Martin_2")



dataset = libs_dataset.DatasetMagnetometer2(folders_list, width = 512, augmentations_count = 1, testing_ratio = 20)

libs.ExportTest(dataset, 64, ModelReference, ModelTestingInt8)