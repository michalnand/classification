import numpy
import torch
import matplotlib.pyplot as plt
from .class_balancer import *
from .class_stats import *

class DatasetMagnetometer2:

    def __init__(self, folders_list, width = 512, augmentations_count = 20, testing_ratio = 10):
        self.width         = width
        self.channels      = 4
        self.input_shape   = (self.channels, self.width)

        self.augmentations_count = augmentations_count
        self.testing_ratio      = int(testing_ratio)

        self.idx_offset_noise   = 32 
        self.level_white_noise  = 0.1 
        self.level_offset_noise = 0.2
        self.scale_min          = 0.5
        self.scale_max          = 1.5
        self.rotation_max       = 20

        self.classes_count = 5
        self.output_shape  = (self.classes_count, )

        self.training_x = []
        self.training_y = []
        self.testing_x  = []
        self.testing_y  = []

        self.class_stats = ClassStats(self.classes_count)

        self.data_idx       = 0
        self.training_count = 0
        self.testing_count  = 0
        for folder in folders_list:
            training_count, testing_count = self._count(folder)
            self.training_count+= training_count
            self.testing_count+= testing_count

              
        self.training_x = numpy.zeros((self.training_count, ) + self.input_shape, dtype=numpy.float32)
        self.training_y = numpy.zeros((self.training_count, ) + self.output_shape, dtype=numpy.float32)
        self.testing_x  = numpy.zeros((self.testing_count, ) + self.input_shape, dtype=numpy.float32)
        self.testing_y  = numpy.zeros((self.testing_count, ) + self.output_shape, dtype=numpy.float32)

        self.training_idx = 0
        self.testing_idx  = 0
        self.data_idx     = 0
        for folder in folders_list:
            self._load(folder)

        '''
        print("training_count = ", self.training_count)
        print("testing_count  = ", self.testing_count)
        print("real training_count = ", self.training_idx)
        print("real testing_count  = ", self.testing_idx)
        '''

        '''
        for i in range(len(self.training_x)):
            mean = numpy.mean(self.training_x[i])
            std  = numpy.std(self.training_x[i])

            self.training_x[i] = (self.training_x[i] - mean)/std

        for i in range(len(self.testing_x)):
            mean = numpy.mean(self.testing_x[i])
            std  = numpy.std(self.testing_x[i])

            self.testing_x[i] = (self.testing_x[i] - mean)/std
        '''

        self.class_balancer = ClassBalancer(self.training_y, self.classes_count)

        print("\n\n\n\n")
        print("dataset summary : \n")
        print("training_count = ", self.get_training_count())
        print("testing_count  = ", self.get_testing_count())
        print("sequence_length = ", self.width)
        print("classes_count =  ", self.classes_count)
        print("training_x shape ", self.training_x.shape)
        print("training_y shape ", self.training_y.shape)
        print("testing_x shape  ", self.testing_x.shape) 
        print("testing_y shape  ", self.testing_y.shape)
        print("training_mean =  ", self.training_x.mean())
        print("training_std =   ", self.training_x.std())
        print("testing_mean =   ", self.testing_x.mean())
        print("testing_std  =   ", self.testing_x.std())
        
        self.class_stats.print_info()
        print("\n")


    
    def get_training_count(self):
        return len(self.training_x)

    def get_testing_count(self):
        return len(self.testing_x)

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_x, self.training_y, batch_size, ballancer = self.class_balancer)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_x, self.testing_y, batch_size)


    def _get_batch(self, x, y, batch_size = 32, ballancer = None):
        result_x = torch.zeros((batch_size, self.channels, self.width))
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size): 
            if ballancer is not None:
                idx = ballancer.get_random_idx()
            else:
                idx = numpy.random.randint(len(x))

            result_x[i]  = torch.from_numpy(x[idx]).float()
            result_y[i]  = torch.from_numpy(y[idx]).float()

        return result_x, result_y

    def _count(self, folder):
        print("loading info from ", folder, end="")
        annotation_data = numpy.genfromtxt(folder + "/anotacie_final.csv", delimiter=';')

        testing_count   = 0
        training_count  = 0
        for i in range(len(annotation_data)-2):
            if self._is_valid(annotation_data, i+1):
                if self._is_testing(self.data_idx):
                    testing_count+= 1
                else:
                    training_count+= 1 + self.augmentations_count
                
                self.data_idx+= 1
                
        print(" items_count = ", training_count, testing_count)
        return training_count, testing_count

    def _load(self, folder):
        print("loading ", folder)
        annotation_data = numpy.genfromtxt(folder + "/anotacie_final.csv", delimiter=';')
        data            = numpy.genfromtxt(folder + "/s0.txt", delimiter=' ', dtype=numpy.float32)

        count      = annotation_data.shape[0]


        #load raw data
        x = numpy.transpose(data)[6]
        y = numpy.transpose(data)[7]
        z = numpy.transpose(data)[8]

    
        for i in range(len(annotation_data)-2):
            if self._is_valid(annotation_data, i+1):

                annotation  = int(annotation_data[i+1][0])
                idx         = int(annotation_data[i+1][3])

                annotation = self._get_class_id(annotation)

                #determine testing or training 
                if self._is_testing(self.data_idx):
                    #create input sample, no augmentation
                    input               = self._create_sample(idx, x, y, z, False)

                    #create one hot encoding target
                    target              = numpy.zeros(self.classes_count)
                    target[annotation]  = 1.0

                    if self.testing_idx < self.testing_count:
                        self.testing_x[self.testing_idx] = input.copy().astype(numpy.float32)
                        self.testing_y[self.testing_idx] = target.copy().astype(numpy.float32)

                    self.testing_idx+= 1
                    
                    self.class_stats.add(target)
                else:
                    #create input sample, no augmentation - keep one sample original
                    input               = self._create_sample(idx, x, y, z, False)

                    #create one hot encoding target
                    target              = numpy.zeros(self.classes_count)
                    target[annotation]  = 1.0

                    if self.training_idx < self.training_count:
                        self.training_x[self.training_idx] = input.copy().astype(numpy.float32)
                        self.training_y[self.training_idx] = target.copy().astype(numpy.float32)

                    self.training_idx+= 1

                    self.class_stats.add(target)


                    #create samples with augmentation
                    for a in range(self.augmentations_count):
                        input               = self._create_sample(idx, x, y, z, True)

                        if self.training_idx < self.training_count:
                            self.training_x[self.training_idx] = input.copy().astype(numpy.float32)
                            self.training_y[self.training_idx] = target.copy().astype(numpy.float32)

                        self.training_idx+= 1

                self.data_idx+= 1
            


    def _create_sample(self, idx, x, y, z, augmentation = False):
        #take random offset in time
        if augmentation:
            idx_offset = int(numpy.random.rand()*self.idx_offset_noise - self.idx_offset_noise//2 + idx)
        else:
            idx_offset = idx

        start = idx_offset - self.width//2
        end   = idx_offset + self.width//2

        if start < 0:
            start = 0
            end   = self.width
        
        if end >= len(x):
            end     = len(x)-1
            start   = end - self.width

        input = numpy.zeros((self.channels, self.width))

        #select sequence
        xs = x[start:end]
        ys = y[start:end]
        zs = z[start:end]

       

        #process augmentation if necessary
        if augmentation:
            #axis noise
            x_noised = self._augmentation_noise(xs)
            y_noised = self._augmentation_noise(ys)
            z_noised = self._augmentation_noise(zs)

            #axis rotation
            x_noised, y_noised, z_noised = self._augmentation_rotation(x_noised, y_noised, z_noised)
 
            #normalise
            '''
            x_normalised = self._standardize(x_noised)
            y_normalised = self._standardize(y_noised)
            z_normalised = self._standardize(z_noised)
            '''

            x_normalised = self._normalise(x_noised, -1.0, 1.0)
            y_normalised = self._normalise(y_noised, -1.0, 1.0)
            z_normalised = self._normalise(z_noised, -1.0, 1.0)
            
            input[0] = x_normalised.copy()
            input[1] = y_normalised.copy()
            input[2] = z_normalised.copy()
        else:
            #normalise
            '''
            x_normalised = self._standardize(xs)
            y_normalised = self._standardize(ys)
            z_normalised = self._standardize(zs)
            '''

            x_normalised = self._normalise(xs, -1.0, 1.0)
            y_normalised = self._normalise(ys, -1.0, 1.0)
            z_normalised = self._normalise(zs, -1.0, 1.0)

            input[0] = x_normalised.copy() 
            input[1] = y_normalised.copy()
            input[2] = z_normalised.copy()

        return input

    def _is_valid(self, annotation_data, idx):
        if numpy.isnan(annotation_data[idx][3]) == True:
            return False
        
        return True

    def _is_testing(self, idx):
        tmp = idx%100
        if tmp < self.testing_ratio:
            return True
        
        return False
               


    def _augmentation_noise(self, x):
        
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

        return result

    def _augmentation_rotation(self, x, y, z):
        input = numpy.array([x, y, z])

        yaw   = self._rnd(-self.rotation_max, self.rotation_max)*numpy.pi/180.0
        pitch = self._rnd(-self.rotation_max, self.rotation_max)*numpy.pi/180.0
        roll  = self._rnd(-self.rotation_max, self.rotation_max)*numpy.pi/180.0

        r = self._rotation_matrix(yaw, pitch, roll)

        result = numpy.matmul(r, input)

        return result[0], result[1], result[2]

    def _rotation_matrix(self, yaw, pitch, roll):
        rx = numpy.zeros((3, 3))
        rx[0][0] = 1.0
        rx[1][1] = numpy.cos(yaw)
        rx[1][2] = -numpy.sin(yaw)
        rx[2][1] =  numpy.sin(yaw)
        rx[2][2] =  numpy.cos(yaw)


        ry = numpy.zeros((3, 3))
        ry[0][0] = numpy.cos(pitch)
        ry[0][2] = numpy.sin(pitch)
        ry[1][1] = 1.0
        ry[2][0] = -numpy.sin(pitch)
        ry[2][2] = numpy.cos(pitch)

        rz = numpy.zeros((3, 3))
        rz[0][0] = numpy.cos(roll)
        rz[0][1] = -numpy.sin(roll)
        rz[1][0] = numpy.sin(roll)
        rz[1][1] = numpy.cos(roll)
        rz[2][2] = 1.0

        result = numpy.matmul(numpy.matmul(rz, ry), rx)

        return result

    def _rnd(self, min, max):
        return numpy.random.rand()*(max - min) + min


    def _get_class_id(self, raw_id):
        class_dict = {}


        class_dict[0]   = 1     #motocycle
        class_dict[1]   = 1     #car
        class_dict[2]   = 2     #supply 3.5t max
        class_dict[3]   = 3     #truck 3.5t+, 13m
        class_dict[4]   = 4     #heavy truck
        class_dict[5]   = 0     #nothing
        class_dict[6]   = 0     #nothing
        class_dict[7]   = 0     #other
        class_dict[-1]  = 0     #other

        
        return class_dict[raw_id]


    def _normalise(self, x, min_value, max_value):
        min_s = numpy.min(x)
        max_s = numpy.max(x)

        if max_s > min_s:
            k = (max_value - min_value)/(max_s - min_s)
            q = max_value - k*max_s
        else:
            k = 0.0
            q = 0.0

        return k*x + q

    def _standardize(self, x):
        return (x - x.mean())/(x.std() + 0.00000001)
