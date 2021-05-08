import numpy
import torch 

class DatasetContrastive:
    def __init__(self, dataset):
        self.dataset        = dataset

        self.classes_count  = 2
        self.input_shape    = self.dataset.input_shape
        self.output_shape   = ( )

        self.training_classes_mapping = []
        for i in range(self.dataset.classes_count):
            self.training_classes_mapping.append([])

        self.testing_classes_mapping = []
        for i in range(self.dataset.classes_count):
            self.testing_classes_mapping.append([])

        for i in range(self.get_training_count()):
            class_idx = numpy.argmax(self.dataset.training_y[i])
            self.training_classes_mapping[class_idx].append(i)

        for i in range(self.get_testing_count()):
            class_idx = numpy.argmax(self.dataset.testing_y[i])
            self.testing_classes_mapping[class_idx].append(i)

        print("\n\n\n\n")
        print("dataset summary - contrastive: \n")
        print("training_count   = ", self.get_training_count())
        print("testing_count    = ", self.get_testing_count())
        print("classes_count    = ", self.classes_count)
        print("input_shape      = ", self.input_shape)
        print("output_shape     = ", self.output_shape)
        print("\n")

    def get_training_count(self):
        return self.dataset.get_training_count()

    def get_testing_count(self):
        return self.dataset.get_testing_count()

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.dataset.training_x, self.dataset.training_y, self.training_classes_mapping, batch_size)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.dataset.testing_x, self.dataset.testing_y, self.testing_classes_mapping, batch_size)

    def _get_batch(self, x, y, classes_mapping, batch_size = 32):
        result_x = torch.zeros((batch_size, 2)  + self.input_shape)
        result_y = torch.zeros((batch_size, ))

        for i in range(batch_size):
            idx_a, idx_b, target = self._get_idx(classes_mapping)
           
            result_x[i][0]  = torch.from_numpy(x[idx_a]).float()
            result_x[i][1]  = torch.from_numpy(x[idx_b]).float()
            result_y[i]     = 1.0*target


        return result_x, result_y

    def _get_idx(self, classes_mapping):
        class_a_idx = numpy.random.randint(self.dataset.classes_count)
        class_b_idx = class_a_idx
        target      = 0

        if numpy.random.randint(2) == 1:
            while class_b_idx == class_a_idx:
                class_b_idx = numpy.random.randint(self.dataset.classes_count)
            target      = 1
            
        item_a_idx = numpy.random.randint(len(classes_mapping[class_a_idx]))
        item_b_idx = numpy.random.randint(len(classes_mapping[class_b_idx]))

        return classes_mapping[class_a_idx][item_a_idx], classes_mapping[class_b_idx][item_b_idx], target


