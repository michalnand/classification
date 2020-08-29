import numpy

class ClassStats:
    def __init__(self, classes_count):
       self.data = numpy.zeros(classes_count)

    def add(self, target):
        idx = numpy.argmax(target)
        self.data[idx]+= 1

    def print_info(self):
        relative = 100.0*self.data/numpy.sum(self.data)

        print("\n")
        print("class stats : ")
        print("class\t\tcount\t\trelative[%]")
        for i in range(len(self.data)):
            print(i, "\t\t", int(self.data[i]), "\t\t", round(relative[i], 2))
        