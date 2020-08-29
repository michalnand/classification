import sys
sys.path.insert(0,'..')

import libs
import libs_dataset
import models.magnetometer_net_0.model as Model

training_data_files_list    = []
training_categories_ids     = []

testing_data_files_list     = []
testing_categories_ids      = []


dataset_path = "/Users/michal/dataset/car_detection/"

training_data_files_list.append(dataset_path + "dataS1RawWinCat1.csv")
training_categories_ids.append(0)

training_data_files_list.append(dataset_path + "dataS1RawWinCat2.csv")
training_categories_ids.append(1)

training_data_files_list.append(dataset_path + "dataS1RawWinCat3.csv")
training_categories_ids.append(2)

training_data_files_list.append(dataset_path + "dataS1RawWinCat4.csv")
training_categories_ids.append(3)

training_data_files_list.append(dataset_path + "dataS1RawWinCatTrailer.csv")
training_categories_ids.append(4)


testing_data_files_list.append(dataset_path + "dataS2RawWinCat1.csv")
testing_categories_ids.append(0)

testing_data_files_list.append(dataset_path + "dataS2RawWinCat2.csv")
testing_categories_ids.append(1)

testing_data_files_list.append(dataset_path + "dataS2RawWinCat3.csv")
testing_categories_ids.append(2)

testing_data_files_list.append(dataset_path + "dataS2RawWinCat4.csv")
testing_categories_ids.append(3)

testing_data_files_list.append(dataset_path + "dataS2RawWinCatTrailer.csv")
testing_categories_ids.append(4)



dataset = libs_dataset.DatasetMagnetometer(training_data_files_list, training_categories_ids, testing_data_files_list, testing_categories_ids)



learning_rates  = [0.0001, 0.0001, 0.0001, 0.00001, 0.00001]
train = libs.Train(dataset, Model, batch_size = 64, learning_rates = learning_rates)


train.step_epochs(100)