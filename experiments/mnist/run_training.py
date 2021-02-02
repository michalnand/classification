import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset

import models.model_1.model as Model1
import models.model_2.model as Model2

#train 30 epochs
epoch_count     = 20 

#use cyclic learning rate
learning_rates  = [0.001, 0.0001, 0.0001]

dataset = libs_dataset.DatasetMnist()

train = libs.Train(dataset, Model1, libs.MetricsClassification, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_1")

train = libs.Train(dataset, Model2, libs.MetricsClassification, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_2")
