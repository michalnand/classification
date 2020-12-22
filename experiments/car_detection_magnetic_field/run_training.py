import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset
import models.net_0.model as Model0
import models.net_1.model as Model1
import models.net_2.model as Model2
import models.net_3.model as Model3
import models.net_4.model as Model4
import models.net_5.model as Model5


#dataset_path = "/Users/michal/dataset/car_detection_2/"
dataset_path = "/home/michal/dataset/car_detection_2/"

folders_list = []
 
folders_list.append(dataset_path + "/Meranie_20_06_03-Kinekus/1")
folders_list.append(dataset_path + "/Meranie_20_06_03-Kinekus/2")
folders_list.append(dataset_path + "/Porubka_03_06_2020")
folders_list.append(dataset_path + "/Meranie_20_05_22-Pribovce_xyz")
folders_list.append(dataset_path + "/Meranie_20_06_01-Lietavska_Lucka/01")
folders_list.append(dataset_path + "/Meranie_20_06_01-Lietavska_Lucka/02")
folders_list.append(dataset_path + "/Bytca")
folders_list.append(dataset_path + "/Kysuce")
folders_list.append(dataset_path + "/Martin_1")
folders_list.append(dataset_path + "/Martin_2")


#window size = 512
#50x artifical data for training
#20% of data for testing
dataset = libs_dataset.DatasetMagnetometer2(folders_list, width = 512, augmentations_count = 50, testing_ratio = 20)
#dataset = libs_dataset.DatasetMagnetometer2(folders_list, width = 512, augmentations_count = 2, testing_ratio = 20)

epoch_count = 200 
learning_rates  = [0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001]

train = libs.Train(dataset, Model4, batch_size = 128, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_4")


'''
train = libs.Train(dataset, Model0, batch_size = 128, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_0")

train = libs.Train(dataset, Model1, batch_size = 128, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_1")

train = libs.Train(dataset, Model2, batch_size = 128, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_2")

train = libs.Train(dataset, Model3, batch_size = 128, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_3")

train = libs.Train(dataset, Model5, batch_size = 128, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_5")
'''