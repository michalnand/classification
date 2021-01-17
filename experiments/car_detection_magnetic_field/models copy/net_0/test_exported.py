import sys
sys.path.insert(0,'../../../..')

import libs
import libs_dataset

import model         as ModelReference
import model_testing as ModelTesting



dataset_path = "/Users/michal/dataset/car_detection_2/"
#dataset_path = "/home/michal/dataset/car_detection_2/"

folders_list = []

'''
folders_list.append(dataset_path + "/Meranie_20_06_03-Kinekus/1")
folders_list.append(dataset_path + "/Meranie_20_06_03-Kinekus/2")
folders_list.append(dataset_path + "/Porubka_03_06_2020")
folders_list.append(dataset_path + "/Meranie_20_05_22-Pribovce_xyz")
folders_list.append(dataset_path + "/Meranie_20_06_01-Lietavska_Lucka/01")
folders_list.append(dataset_path + "/Meranie_20_06_01-Lietavska_Lucka/02")
'''

folders_list.append(dataset_path + "/Bytca")
folders_list.append(dataset_path + "/Kysuce")
folders_list.append(dataset_path + "/Martin_1")
folders_list.append(dataset_path + "/Martin_2")

dataset = libs_dataset.DatasetMagnetometer2(folders_list, width = 512, augmentations_count = 1, testing_ratio = 20)


libs.ExportTest(dataset, 64, ModelReference, ModelTesting)