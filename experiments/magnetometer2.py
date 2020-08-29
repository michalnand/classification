import sys
sys.path.insert(0,'..')

import libs
import libs_dataset
import models.magnetometer_net_0.model as Model0
import models.magnetometer_net_1.model as Model1
import models.magnetometer_net_2.model as Model2
import models.magnetometer_net_3.model as Model3


folders_list = []
folders_list.append("/Users/michal/dataset/car_detection_2/Meranie_20_06_03-Kinekus/1")
folders_list.append("/Users/michal/dataset/car_detection_2/Meranie_20_06_03-Kinekus/2")
folders_list.append("/Users/michal/dataset/car_detection_2/Porubka_03_06_2020")
folders_list.append("/Users/michal/dataset/car_detection_2/Meranie_20_05_22-Pribovce_xyz")
folders_list.append("/Users/michal/dataset/car_detection_2/Meranie_20_06_01-Lietavska_Lucka/01")


dataset = libs_dataset.DatasetMagnetometer2(folders_list, width = 512, augmentations_count = 40, testing_ratio = 0.2)

epoch_count = 100
learning_rates  = [0.0001, 0.0001, 0.0001, 0.00001, 0.00001]


train = libs.Train(dataset, Model0, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "../models/magnetometer_net_0")

train = libs.Train(dataset, Model1, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "../models/magnetometer_net_1")

train = libs.Train(dataset, Model2, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "../models/magnetometer_net_2")

train = libs.Train(dataset, Model3, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "../models/magnetometer_net_3")