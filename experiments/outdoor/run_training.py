import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset

import models.model_0.model as Model0
import models.model_1.model as Model1


#train epochs
epoch_count     = 1000 

#use cyclic learning rate
learning_rates  = [0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001]

folders_training = []
#folders_training.append("/Users/michal/dataset/outdoor/lietavska_lucka/")
#folders_training.append("/Users/michal/dataset/outdoor/istrobotics_0/")
#folders_training.append("/Users/michal/dataset/outdoor/istrobotics_1/")
#folders_training.append("/Users/michal/dataset/outdoor/istrobotics_2/")

folders_training.append("/home/michal/dataset/outdoor/lietavska_lucka/")
folders_training.append("/home/michal/dataset/outdoor/istrobotics_0/")
folders_training.append("/home/michal/dataset/outdoor/istrobotics_1/")
folders_training.append("/home/michal/dataset/outdoor/istrobotics_2/")
 

classes_ids     = [8, 12, 21, 22, 23]

dataset = libs_dataset.DatasetSegmentation(folders_training, folders_training, classes_ids)

train = libs.Train(dataset, Model1, libs.MetricsSegmentation, batch_size = 8, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_1")

train = libs.Train(dataset, Model0, libs.MetricsSegmentation, batch_size = 8, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_0")
