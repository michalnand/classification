import numpy
import torch
import matplotlib.pyplot as plt
from .class_balancer import *
from .class_stats import *

class DatasetMagnetometer2:

    def __init__(self, folders_list, width = 512, augmentations_count = 20, testing_ratio = 0.1):
        self.width         = width
        self.channels      = 4
        self.input_shape   = (self.channels, self.width)

        self.augmentations_count = augmentations_count
        self.testing_ratio      = testing_ratio

        self.idx_offset_noise   = 32 
        self.level_white_noise  = 0.1 
        self.level_offset_noise = 0.8
        self.scale_min          = 0.5
        self.scale_max          = 1.5

        self.classes_count = 4
        self.output_shape  = (self.classes_count, )

        self.training_x = []
        self.training_y = []
        self.testing_x  = []
        self.testing_y  = []

        self.class_stats = ClassStats(self.classes_count)

        for folder in folders_list:
            self._load(folder)

        self.training_x = numpy.array(self.training_x)
        self.training_y = numpy.array(self.training_y)
        self.testing_x = numpy.array(self.testing_x)
        self.testing_y = numpy.array(self.testing_y)


        for i in range(len(self.training_x)):
            mean = numpy.mean(self.training_x[i])
            std  = numpy.std(self.training_x[i])

            self.training_x[i] = (self.training_x[i] - mean)/std

        for i in range(len(self.testing_x)):
            mean = numpy.mean(self.testing_x[i])
            std  = numpy.std(self.testing_x[i])

            self.testing_x[i] = (self.testing_x[i] - mean)/std
        
        self.class_balancer = ClassBalancer(self.training_y, self.classes_count)

        print("\n")
        print("training_count = ", self.get_training_count())
        print("testing_count  = ", self.get_testing_count())
        print("sequence_length  = ", self.width)
        print("classes_count = ", self.classes_count)
        print("training_x shape ", self.training_x.shape)
        print("training_y shape ", self.training_y.shape)
        print("testing_x shape ", self.testing_x.shape) 
        print("testing_y shape ", self.testing_y.shape)
        self.class_stats.print_info()
        print("\n")


    
    def get_training_count(self):
        return len(self.training_x)

    def get_testing_count(self):
        return len(self.testing_x)

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_x, self.training_y, ballancer = self.class_balancer)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_x, self.testing_y)


    def _get_batch(self, x, y, batch_size = 32, ballancer = None):
        result_x = torch.zeros((batch_size, self.channels, self.width))
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size): 
            if ballancer is not None:
                idx = ballancer.get_random_idx()
            else:
                idx = numpy.random.randint(len(x))

            result_x[i]  = torch.from_numpy(x[idx])
            result_y[i]  = torch.from_numpy(y[idx])

        return result_x, result_y


    def _load(self, folder):
        print("loading ", folder)
        annotation_data = numpy.genfromtxt(folder + "/anotacie_final.csv", delimiter=';')
        data            = numpy.genfromtxt(folder + "/s0.txt", delimiter=' ')

        count      = annotation_data.shape[0]

        print("items count : ", count)

        #load raw data
        x = numpy.transpose(data)[6]
        y = numpy.transpose(data)[7]
        z = numpy.transpose(data)[8]

    
        #normalise
        x_filtered = (x  - numpy.mean(x))/numpy.std(x)
        y_filtered = (y  - numpy.mean(y))/numpy.std(y)
        z_filtered = (z  - numpy.mean(z))/numpy.std(z) 

        #x_filtered = x - numpy.roll(x, 1)
        #y_filtered = y - numpy.roll(y, 1)
        #z_filtered = z - numpy.roll(z, 1)

        for i in range(len(annotation_data)-1):
            annotation  = int(annotation_data[i+1][0])
            idx         = int(annotation_data[i+1][3])

            annotation = self._get_class_id(annotation)


            #check stream boundaries
            if idx > 2*self.width and idx < len(x) - 2*self.width:
                #random determine testing or training 
                if numpy.random.rand() < self.testing_ratio:
                    #create input sample, no augmentation
                    input               = self._create_sample(idx, x_filtered, y_filtered, z_filtered, False)

                    #create one hot encoding target
                    target              = numpy.zeros(self.classes_count)
                    target[annotation]  = 1.0
        
                    self.testing_x.append(input.copy())
                    self.testing_y.append(target.copy())
                  
                    self.class_stats.add(target)
                else:
                    #create input sample, no augmentation - keep one sample original
                    input               = self._create_sample(idx, x_filtered, y_filtered, z_filtered, False)

                    #create one hot encoding target
                    target              = numpy.zeros(self.classes_count)
                    target[annotation]  = 1.0

                    self.training_x.append(input.copy())
                    self.training_y.append(target.copy()) 
                    
                    self.class_stats.add(target)

                    #create samples with augmentation
                    for a in range(self.augmentations_count):
                        input               = self._create_sample(idx, x_filtered, y_filtered, z_filtered, True)

                        self.training_x.append(input.copy())
                        self.training_y.append(target.copy())

        


    def _create_sample(self, idx, x, y, z, augmentation = False):
        #take random offset in time
        if augmentation:
            idx_offset = int(numpy.random.rand()*self.idx_offset_noise - self.idx_offset_noise//2 + idx)
        else:
            idx_offset = idx

        start = idx_offset - self.width//2
        end   = idx_offset + self.width//2

        input = numpy.zeros((self.channels, self.width))

        #process augmentation if necessary
        if augmentation:
            #axis permutation
            #indices = numpy.random.permutation(3)
            indices = [0, 1, 2]
 
            input[indices[0]] = self._augmentation(x[start:end]).copy()
            input[indices[1]] = self._augmentation(y[start:end]).copy()
            input[indices[2]] = self._augmentation(z[start:end]).copy()
        else:
            input[0] = x[start:end].copy()
            input[1] = y[start:end].copy()
            input[2] = z[start:end].copy()

        return input


    def _augmentation(self, x):
        
        min = numpy.min(x)
        max = numpy.max(x)

        #random DC offset
        offset_noise = self._rnd(min, max)

        #random white noise
        white_noise  = numpy.random.rand(len(x))*(max - min) + min

        #random scale
        scale = self._rnd(self.scale_min, self.scale_max)

        #mix all
        result = (1.0 - self.level_white_noise)*scale*x + self.level_white_noise*white_noise
        result = result + self.level_offset_noise*offset_noise

        '''
        if numpy.random.randint(2) == 0:
            result = -1.0*result
        '''

        return result

    def _rnd(self, min, max):
        return numpy.random.rand()*(max - min) + min


    def _get_class_id(self, raw_id):
        class_dict = {}

        class_dict[0]   = 1
        class_dict[1]   = 1
        class_dict[2]   = 2
        class_dict[3]   = 3
        class_dict[4]   = 3
        class_dict[5]   = 0
        class_dict[6]   = 0
        class_dict[7]   = 0
        class_dict[-1]  = 0
        

        return class_dict[raw_id]

if __name__ == "__main__":

    folders_list = []
    folders_list.append("/Users/michal/dataset/car_detection_2/Meranie_20_06_03-Kinekus/1")

    dataset = DatasetMagnetometer2(folders_list)

 
    training_x, training_y = dataset.get_training_batch()

    for i in range(10):
        plt.plot(training_x[i][0])
        plt.plot(training_x[i][1])
        plt.plot(training_x[i][2])
        plt.show()