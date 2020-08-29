import numpy

class ClassBalancer:
    def __init__(self, labels, classes_count):
        self.indices = [ [] for i in range(classes_count) ]

        for i in range(len(labels)):
            class_idx = numpy.argmax(labels[i])
            self.indices[class_idx].append(i)

    def get_random_idx(self):
        
        class_idx = numpy.random.randint(len(self.indices))
        while len(self.indices[class_idx]) == 0:
            class_idx = numpy.random.randint(len(self.indices))
           
        idx = numpy.random.randint(len(self.indices[class_idx]))

        return self.indices[class_idx][idx]
