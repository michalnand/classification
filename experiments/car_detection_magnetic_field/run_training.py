import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset
import models.net_0.model as Model0
import models.net_1.model as Model1
import models.net_2.model as Model2
import models.net_3.model as Model3


dataset_path = "/Users/michal/dataset/"
folders_list = []

folders_list.append(dataset_path + "/car_detection_2/Meranie_20_06_03-Kinekus/1")
folders_list.append(dataset_path + "/car_detection_2/Meranie_20_06_03-Kinekus/2")
folders_list.append(dataset_path + "/car_detection_2/Porubka_03_06_2020")
folders_list.append(dataset_path + "/car_detection_2/Meranie_20_05_22-Pribovce_xyz")
folders_list.append(dataset_path + "/car_detection_2/Meranie_20_06_01-Lietavska_Lucka/01")

dataset = libs_dataset.DatasetMagnetometer2(folders_list, width = 512, augmentations_count = 10, testing_ratio = 0.2)

epoch_count = 200
learning_rates  = [0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001]

 
train = libs.Train(dataset, Model0, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_0")

train = libs.Train(dataset, Model1, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_1")

train = libs.Train(dataset, Model2, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_2")

train = libs.Train(dataset, Model3, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_3")