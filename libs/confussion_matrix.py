import numpy


class ConfussionMatrix:
    def __init__(self, classes_count):
        self.classes_count = classes_count
        self.clear()

    def clear(self):
        self.confussion_matrix = numpy.zeros((self.classes_count, self.classes_count), dtype=int)
        
        self.hit_count      = 0
        self.miss_count     = 0
        self.total_count    = 0
        self.accuracy       = 0

        self.class_accuracy = numpy.zeros(self.classes_count)


    def add_batch(self, target, predicted):
        for i in range(len(target)):
            self.add(target[i], predicted[i])

    def add(self, target, predicted):
        target_idx      = numpy.argmax(target)
        predicted_idx   = numpy.argmax(predicted)

        self.confussion_matrix[predicted_idx][target_idx]+= 1

    def compute(self):
        self.total_count = numpy.sum(self.confussion_matrix)
        self.hit_count   = 0
        for i in range(self.classes_count):
            self.hit_count+= self.confussion_matrix[i][i]

        self.miss_count = self.total_count - self.hit_count
        self.accuracy  = self.hit_count*100.0/self.total_count

        class_hit = numpy.zeros(self.classes_count, dtype=int)
        for i in range(self.classes_count):
            class_hit[i] = self.confussion_matrix[i][i]

        class_count = numpy.ones(self.classes_count, dtype=int)
        for target_idx in range(self.classes_count):
            for predicted_idx in range(self.classes_count):
                class_count[target_idx]+= self.confussion_matrix[predicted_idx][target_idx]

        self.class_accuracy = class_hit*100.0/class_count


    def get_result(self): 
        result_str = ""
        result_str+= "accuracy   = " + str(round(self.accuracy, 3))  + " [%]" + "\n"
        result_str+= "hit_count  = " + str(self.hit_count)  + "\n"
        result_str+= "miss_count = " + str(self.miss_count) + "\n"
        result_str+= "\n"

        result_str+= "class_accuracy = "
        for target_idx in range(self.classes_count):
            result_str+= str(round(self.class_accuracy[target_idx], 3)) + "%   "
        result_str+= "\n"

        result_str+= "\n\n"
        result_str+= "confusion_matrix = \n"

        for target_idx in range(self.classes_count):
            for predicted_idx in range(self.classes_count):
                 result_str+= str(self.confussion_matrix[target_idx][predicted_idx]) + "\t "
            result_str+= "\n"
        result_str+= "\n"

        return result_str