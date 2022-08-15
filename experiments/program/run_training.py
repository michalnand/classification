import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset

import models.model_0.model as Model

#train 30 epochs
epoch_count     = 100

#use cyclic learning rate
learning_rates  = [0.001] 

dataset = libs_dataset.DatasetProgram()

train = libs.Train(dataset, Model, libs.MetricsClassification, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_0")

