import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset

import models.model_0.model as Model0


#train epochs
epoch_count     = 100 

#use cyclic learning rate
learning_rates  = [0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001]

dataset = libs_dataset.DatasetCityscapes("/Users/michal/dataset/cityscapes/", height=256, width=512)

train = libs.Train(dataset, Model0, libs.MetricsSegmentation, batch_size = 8, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_0")
