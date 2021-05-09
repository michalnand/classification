import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset

import models.model_0.model     as Model0
import models.model_0_c.model   as Model0C

#train 10 epochs
epoch_count     = 10 

#use cyclic learning rate
learning_rates  = [0.001, 0.0001]

#load common classification dataset
dataset_orig = libs_dataset.DatasetMnist()

#convert dataset to two class contrastive dataset
dataset     = libs_dataset.DatasetContrastive(dataset_orig)

'''
#process training, with contrastive loss
train = libs.Train(dataset, Model0, libs.MetricsContrastive, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_0")
'''

#common supervised training, 1 epoch is enough
train = libs.Train(dataset_orig, Model0C, libs.MetricsClassification, batch_size = 64, learning_rates = [0.001])
train.step_epochs(1, log_path = "./models/model_0_c")