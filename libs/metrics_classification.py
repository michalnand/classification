import torch
import numpy

from .confusion_matrix import *

class MetricsClassification:
    def __init__(self, output_shape):
        self.classes_count = numpy.prod(output_shape)

        self.confusion_matrix_training = ConfusionMatrix(self.classes_count)
        self.confusion_matrix_testing  = ConfusionMatrix(self.classes_count)

        self._loss_training = []
        self._loss_testing  = []

    def loss_training(self, target_t, predicted_t):
        return ((target_t.detach() - predicted_t)**2).mean()

    def loss_testing(self, target_t, predicted_t):
        return ((target_t.detach() - predicted_t)**2).mean()

    def add_training(self, target_t, predicted_t):
        loss = self.loss_training(target_t, predicted_t)
        self._loss_training.append(loss.detach().to("cpu").numpy())
        self.confusion_matrix_training.add_batch(target_t.detach().to("cpu").numpy(), predicted_t.detach().to("cpu").numpy())

    def add_testing(self, target_t, predicted_t):
        loss = self.loss_testing(target_t, predicted_t)
        self._loss_testing.append(loss.detach().to("cpu").numpy())
        self.confusion_matrix_testing.add_batch(target_t.detach().to("cpu").numpy(), predicted_t.detach().to("cpu").numpy())


    def compute(self):
        self.confusion_matrix_training.compute()
        self.confusion_matrix_testing.compute()

        self.loss_training_mean     = numpy.array(self._loss_training).mean()
        self.loss_training_std      = numpy.array(self._loss_training).std()
        self.loss_testing_mean      = numpy.array(self._loss_testing).mean()
        self.loss_testing_std       = numpy.array(self._loss_testing).std()

    def get_score(self):
        return self.confusion_matrix_testing.accuracy

    def get_short(self):

        result = ""
        result+= str(self.loss_training_mean) + " "
        result+= str(self.loss_training_std) + " "
        result+= str(self.confusion_matrix_training.accuracy) + " "
        result+= str(self.loss_testing_mean) + " "
        result+= str(self.loss_testing_std) + " "
        result+= str(self.confusion_matrix_testing.accuracy) + " "

        return result

    def get_full(self):

        result = ""
        result+= "TRAINING result\n\n"
        result+= "loss_mean = " + str(self.loss_training_mean) + "\n"
        result+= "loss_std  = " + str(self.loss_training_std) + "\n\n\n"
        result+= self.confusion_matrix_training.get_result()

        result+= "\n\n\n\n"

        result+= "TESTING result\n\n"
        result+= "loss_mean = " + str(self.loss_testing_mean) + "\n"
        result+= "loss_std  = " + str(self.loss_testing_std) + "\n\n\n"
        result+= self.confusion_matrix_testing.get_result()

        return result

