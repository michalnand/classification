import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset

import models.model_0.model as Model0


#train epochs
epoch_count     = 100 

#use cyclic learning rate
learning_rates  = [0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001]

folders_training = []
#folders_training.append("/Users/michal/dataset/outdoor/lietavska_lucka/")
#folders_training.append("/Users/michal/dataset/outdoor/istrobotics_0/")
#folders_training.append("/Users/michal/dataset/outdoor/istrobotics_1/")
#folders_training.append("/Users/michal/dataset/outdoor/istrobotics_2/")
#folders_training.append("/Users/michal/dataset/outdoor/nature/")
#folders_training.append("/Users/michal/dataset/outdoor/za_park/") 
#folders_training.append("/Users/michal/dataset/outdoor/street/") 

folders_training.append("/home/michal/dataset/outdoor/lietavska_lucka/images/")
folders_training.append("/home/michal/dataset/outdoor/istrobotics_0/images/")
folders_training.append("/home/michal/dataset/outdoor/istrobotics_1/images/")
folders_training.append("/home/michal/dataset/outdoor/istrobotics_2/images/")
folders_training.append("/home/michal/dataset/outdoor/nature/images/")
folders_training.append("/home/michal/dataset/outdoor/za_park/images/")
#folders_training.append("/home/michal/dataset/outdoor/street/images/")
 
 
dataset = libs_dataset.DatasetTransformations(folders_training, folders_training, height = 256, width = 256)

train = libs.Train(dataset, Model0, libs.MetricsRegression, batch_size = 32, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/model_0")
