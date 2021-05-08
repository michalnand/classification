import numpy
import torch
import time


from .confusion_matrix import *


class ExportTest:
    def __init__(self, dataset, batch_size, ModelReference, ModelTesting):
        self.dataset    = dataset
        self.batch_size = batch_size

        self.model_reference  = ModelReference.Create(self.dataset.input_shape, self.dataset.output_shape)
        self.model_testing    = ModelTesting.Create(self.dataset.input_shape, self.dataset.output_shape)
        
        self.model_reference.load("trained/")
        self.model_testing.load("trained/")

        reference_confusion_matrix, testing_confusion_matrix = self._process()

        reference_accuracy   = reference_confusion_matrix.accuracy
        testing_accuracy     = testing_confusion_matrix.accuracy

        log_str = "reference_accuracy = " + str(reference_accuracy) + "\n"
        log_str+= "testing_accuracy = " + str(testing_accuracy) + "\n"
        log_str+= "\n\n"
        log_str+= "reference result\n"
        log_str+= reference_confusion_matrix.get_result() + "\n\n\n\n"
        log_str+= "testing result\n"
        log_str+= testing_confusion_matrix.get_result() + "\n\n\n\n"

        print(log_str)
        

    def _process(self):

        batch_count                 = (self.dataset.get_testing_count()+self.batch_size) // self.batch_size
        reference_confusion_matrix = ConfusionMatrix(self.dataset.classes_count)
        testing_confusion_matrix   = ConfusionMatrix(self.dataset.classes_count)

        for batch_id in range(batch_count):
            x, target_y = self.dataset.get_testing_batch(self.batch_size)

            x           = x.to(self.model_reference.device)
            target_y    = target_y.to(self.model_reference.device).detach().to("cpu").numpy()

            reference_y = self.model_reference.forward(x).detach().to("cpu").numpy()
            testing_y   = self.model_testing.forward(x).detach().to("cpu").numpy()

            if len(target_y.shape) > 2:
                target_y    = numpy.reshape(target_y,  (target_y.shape[0]*target_y.shape[1], target_y.shape[2]))
                reference_y = numpy.reshape(reference_y,  (reference_y.shape[0]*reference_y.shape[1], reference_y.shape[2]))
                testing_y   = numpy.reshape(testing_y, (testing_y.shape[0]*testing_y.shape[1], testing_y.shape[2]))
            
            reference_confusion_matrix.add_batch(target_y, reference_y)
            testing_confusion_matrix.add_batch(target_y, testing_y)

        reference_confusion_matrix.compute()
        testing_confusion_matrix.compute()

        return reference_confusion_matrix, testing_confusion_matrix