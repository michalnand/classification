import numpy
import torch 

class DatasetAutoencoder:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_training_count(self):
        return self.dataset.get_training_count()

    def get_testing_count(self):
        return self.dataset.get_testing_count()

    def get_training_batch(self, batch_size = 32):
        x, y = self.dataset.get_training_batch(batch_size=batch_size)
        return x, x

    def get_testing_batch(self, batch_size = 32):
         x, y = self.dataset.get_testing_batch(batch_size=batch_size)
        return x, x
