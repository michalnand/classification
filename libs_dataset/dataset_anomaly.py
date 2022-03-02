import numpy
import torch 

class DatasetAnomaly:
    def __init__(self, dataset):
        self.dataset        = dataset

        self.input_shape    = self.dataset.input_shape
        self.classes_count  = self.dataset.output_shape[0]

        self.output_shape   = (self.classes_count, )


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
        print("dataset summary - anomaly: \n")
        print("training_count   = ", self.get_training_count())
        print("testing_count    = ", self.get_testing_count())
        print("input_shape      = ", self.input_shape)
        print("output_shape     = ", self.output_shape)
        print("\n")

    def get_training_count(self):
        return self.dataset.get_training_count()

    def get_testing_count(self):
        return self.dataset.get_testing_count()

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.dataset.training_x, self.training_classes_mapping, batch_size)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.dataset.testing_x, self.testing_classes_mapping, batch_size)


    def _get_batch(self, x, classes_mapping, batch_size):
        result_x = torch.zeros((batch_size, 2, )  + self.input_shape)
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size):
            class_id = numpy.random.randint(self.dataset.classes_count)

            idx_a = numpy.random.randint(len(classes_mapping[class_id]))
            idx_b = numpy.random.randint(len(classes_mapping[class_id]))

            idx_a = classes_mapping[class_id][idx_a]
            idx_b = classes_mapping[class_id][idx_b]
            
            result_x[i][0]  = torch.from_numpy(x[idx_a]).float()
            result_x[i][1]  = torch.from_numpy(x[idx_b]).float()
            result_y[i][class_id] = 1.0

        return result_x, result_y

