import numpy
import matplotlib.pyplot as plt
import torch
from .class_balancer import *


class DatasetMagnetometer:

    def __init__(self, training_data_files_list, training_categories_ids, testing_data_files_list, testing_categories_ids):
        self.classes_count = 1 + numpy.max(training_categories_ids)
        self.output_shape  = (self.classes_count, )

        self.training_x, self.training_y = self._load(training_data_files_list, training_categories_ids, augmentation = 50)
        self.testing_x,  self.testing_y  = self._load(testing_data_files_list, testing_categories_ids)

        for i in range(len(self.training_x)):
            mean = numpy.mean(self.training_x[i])
            std  = numpy.std(self.training_x[i])

            self.training_x[i] = (self.training_x[i] - mean)/std

        for i in range(len(self.testing_x)):
            mean = numpy.mean(self.testing_x[i])
            std  = numpy.std(self.testing_x[i])

            self.testing_x[i] = (self.testing_x[i] - mean)/std

        self.width    = len(self.training_x[0])

        self.input_shape = (1, self.width)


        self.class_balancer = ClassBalancer(self.training_y, self.classes_count)

        print("\n")
        print("training_count = ", self.get_training_count())
        print("testing_count  = ", self.get_testing_count())
        print("sequence_length  = ", self.width)
        print("classes_count = ", self.classes_count)
        print("\n")

        
        #plt.plot(self.training_x[100])
        #plt.show()
    
    def get_training_count(self):
        return len(self.training_x)

    def get_testing_count(self):
        return len(self.testing_x)

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_x, self.training_y, ballancer = self.class_balancer)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_x, self.testing_y)

    def _load(self, data_files, categories_ids, augmentation = 0):
        data = numpy.genfromtxt(data_files[0], delimiter=',')
        count      = data.shape[0]
        width      = data.shape[1]

        result_x = []
        result_y = []

        input_sequence_length      = 340

        target_sequence_length = 256

        index_start              = (input_sequence_length - target_sequence_length)//2
        index_end                = input_sequence_length - index_start


        for j in range(len(data_files)):
            file_name   = data_files[j]
            print("loading ", file_name)
            
            data        = numpy.genfromtxt(file_name, delimiter=',')

            if data.shape[1] > input_sequence_length:
                data = numpy.delete(data, input_sequence_length, axis=1)


            cat         = numpy.zeros(self.classes_count)
            cat_id      = categories_ids[j]
            cat[cat_id] = 1.0

            
            
            for i in range(len(data)):
                hp_filtered = data[i]  - numpy.roll(data[i], 1)
                hp_filtered[0] = 0.0
                hp_filtered[1] = 0.0
                hp_filtered[2] = 0.0

                cutted_sequence = hp_filtered[index_start:index_end]

                result_x.append(cutted_sequence)
                result_y.append(cat)

                if augmentation > 0:
                    for k in range(augmentation):
                        output_ = self._augmentation(hp_filtered, target_sequence_length=target_sequence_length)

                        result_x.append(output_)
                        result_y.append(cat)

        result_x = numpy.asarray(result_x)
        result_y = numpy.asarray(result_y)  

        return result_x, result_y


    def _get_batch(self, x, y, batch_size = 32, ballancer = None):
        result_x = torch.zeros((batch_size, 1, self.width))
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size):
            if ballancer is not None:
                idx = ballancer.get_random_idx()
            else:
                idx = numpy.random.randint(len(x))

            result_x[i][0] = torch.from_numpy(x[idx])
            result_y[i] = torch.from_numpy(y[idx])

        return result_x, result_y



    def _augmentation(self, input, level_white_noise = 0.1, level_offset_noise = 0.5, target_sequence_length = 256):
        
        index_start              = numpy.random.randint(len(input) - target_sequence_length)
        index_end                = index_start + target_sequence_length

        cutted_input = input[index_start:index_end]


        min = numpy.min(cutted_input)
        max = numpy.max(cutted_input)

        offset_noise = numpy.random.rand()*(max - min) + min
        white_noise  = numpy.random.rand(len(cutted_input))*(max - min) + min

        scale = numpy.random.rand()*(2.0 - 0.5) + 0.5

        result = (1.0 - level_white_noise)*scale*cutted_input + level_white_noise*white_noise
        result = result + level_offset_noise*offset_noise

        return result

