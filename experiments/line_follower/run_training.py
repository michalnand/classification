import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset
import models.net_0.model as Model0
import models.net_1.model as Model1

epoch_count     = 30 
learning_rates  = [0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001]


dataset = libs_dataset.DatasetLineFollower(width = 8, height = 8, classes_count = 5, training_count = 50000, testing_count = 5000)
#train = libs.Train(dataset, Model0, libs.MetricsClassification, batch_size = 64, learning_rates = learning_rates)
#train.step_epochs(epoch_count, log_path = "./models/net_0")


train = libs.Train(dataset, Model1, libs.MetricsClassification, batch_size = 64, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "./models/net_1")
