import numpy
import torch
import torchvision

class DatasetMnist:

    def __init__(self):

        train = torchvision.datasets.MNIST("./files/", train=True, download=True)
        test  = torchvision.datasets.MNIST("./files/", train=False, download=True)

        self.training_count = len(train)
        self.testing_count  = len(test)

        
        
        self.channels      = 1
        self.height        = 28
        self.width         = 28
        self.input_shape   = (self.channels, self.height, self.width)

        self.classes_count = 10
        self.output_shape  = (self.classes_count, )

        self.training_x = []
        self.training_y = []
        self.testing_x  = []
        self.testing_y  = []

        
        self.training_x = numpy.zeros((self.training_count, ) + self.input_shape, dtype=numpy.float32)
        self.training_y = numpy.zeros((self.training_count, ) + self.output_shape, dtype=numpy.float32)
        self.testing_x  = numpy.zeros((self.testing_count, ) + self.input_shape, dtype=numpy.float32)
        self.testing_y  = numpy.zeros((self.testing_count, ) + self.output_shape, dtype=numpy.float32)

        idx = 0
        for image, class_id in train:
            image_np = numpy.array(image)
            self.training_x[idx][0]         = image_np
            self.training_y[idx][class_id]  = 1.0
            idx+= 1

        idx = 0
        for image, class_id in test:
            image_np = numpy.array(image)
            self.testing_x[idx][0]         = image_np
            self.testing_y[idx][class_id]  = 1.0
            idx+= 1
            
      
        mean = numpy.mean(self.training_x)
        std  = numpy.std(self.training_x)

        self.training_x = (self.training_x - mean)/std

        mean = numpy.mean(self.testing_x)
        std  = numpy.std(self.testing_x)
 
        self.testing_x = (self.testing_x - mean)/std
        

        print("\n\n\n\n")
        print("dataset summary : \n")
        print("training_count = ", self.get_training_count())
        print("testing_count  = ", self.get_testing_count())
        print("channels = ", self.height)
        print("height   = ", self.height)
        print("width    = ", self.width)
        print("classes_count =  ", self.classes_count)
        print("training_x shape ", self.training_x.shape)
        print("training_y shape ", self.training_y.shape)
        print("testing_x shape  ", self.testing_x.shape) 
        print("testing_y shape  ", self.testing_y.shape)
        print("\n")


    
    def get_training_count(self):
        return len(self.training_x)

    def get_testing_count(self):
        return len(self.testing_x)

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_x, self.training_y, batch_size)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_x, self.testing_y, batch_size)

    def _get_batch(self, x, y, batch_size = 32):
        result_x = torch.zeros((batch_size, self.channels, self.height, self.width))
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size): 
            idx = numpy.random.randint(len(x))

            result_x[i]  = torch.from_numpy(x[idx]).float()
            result_y[i]  = torch.from_numpy(y[idx]).float()

        return result_x, result_y


if __name__ == "__main__":
    dataset = DatasetMnist()