import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset
import models.net_0.model as Model0
import models.net_1.model as Model1


dataset = libs_dataset.DatasetLineFollower(width = 8, height = 8, classes_count = 5, training_count = 60000, testing_count = 6000)



epoch_count = 100 
learning_rates  = [0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001]


train = libs.Train(dataset, Model0, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_0")

train = libs.Train(dataset, Model1, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_1")