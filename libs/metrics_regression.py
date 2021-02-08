import torch
import numpy

class MetricsRegression:
    def __init__(self, output_shape):
        self.classes_count = numpy.prod(output_shape)

        self._loss_training = []
        self._loss_testing  = []

    def loss_training(self, target_t, predicted_t):
        return ((target_t.detach() - predicted_t)**2).mean()

    def loss_testing(self, target_t, predicted_t):
        return ((target_t.detach() - predicted_t)**2).mean()

    def add_training(self, target_t, predicted_t):
        loss = self.loss_training(target_t, predicted_t)
        self._loss_training.append(loss.detach().to("cpu").numpy())

    def add_testing(self, target_t, predicted_t):
        loss = self.loss_testing(target_t, predicted_t)
        self._loss_testing.append(loss.detach().to("cpu").numpy())


    def compute(self):
        self.loss_training_mean     = numpy.array(self._loss_training).mean()
        self.loss_training_std      = numpy.array(self._loss_training).std()
        self.loss_training_min      = numpy.array(self._loss_training).min()
        self.loss_training_max      = numpy.array(self._loss_training).max()
        

        self.loss_testing_mean      = numpy.array(self._loss_testing).mean()
        self.loss_testing_std       = numpy.array(self._loss_testing).std()
        self.loss_testing_min       = numpy.array(self._loss_testing).min()
        self.loss_testing_max       = numpy.array(self._loss_testing).max()
        

    def get_score(self):
        return -self.loss_testing_mean

    def get_short(self):

        result = ""
        result+= str(self.loss_training_mean) + " "
        result+= str(self.loss_training_std) + " "
        result+= str(self.loss_training_min) + " "
        result+= str(self.loss_training_max) + " "
        
        result+= str(self.loss_testing_mean) + " "
        result+= str(self.loss_testing_std) + " "
        result+= str(self.loss_testing_min) + " "
        result+= str(self.loss_testing_max) + " "

        return result

    def get_full(self):

        result = ""
        result+= "TRAINING result\n\n"
        result+= "loss_mean = " + str(self.loss_training_mean) + "\n"
        result+= "loss_std  = " + str(self.loss_training_std) + "\n"
        result+= "loss_min  = " + str(self.loss_training_min) + "\n"
        result+= "loss_max  = " + str(self.loss_training_max) + "\n"

        result+= "\n"

        result+= "TESTING result\n\n"
        result+= "loss_mean = " + str(self.loss_testing_mean) + "\n"
        result+= "loss_std  = " + str(self.loss_testing_std) + "\n"
        result+= "loss_min  = " + str(self.loss_testing_min) + "\n"
        result+= "loss_max  = " + str(self.loss_testing_max) + "\n"

        return result

