import numpy
import torch
import os.path
from os import path

class StreamMagnetometer:
    def __init__(self, path, width = 512):
        self.width         = width
        self.channels      = 4
        self.input_shape   = (self.channels, self.width)

        self.classes_count = 4
        self.output_shape  = (self.classes_count, )

        self._load(path)
        self._load_stats(path)

    
    def get_window(self, idx):
        return self.time_stamp[idx], self._get_window_data(idx)

    def get_idx_max(self):
        return self.items_count - self.width
    

    def _get_window_data(self, idx):
        result = torch.zeros((1, self.channels, self.width))

        result[0][0] = self.stream_data[0][idx:idx+self.width]
        result[0][1] = self.stream_data[1][idx:idx+self.width]
        result[0][2] = self.stream_data[2][idx:idx+self.width]

        return result


    def _load(self, folder):
        print("loading ", folder)
        data            = numpy.genfromtxt(folder + "/s1.txt", delimiter=' ')

        self.items_count = len(data)

        #extract time
        hh = numpy.transpose(data)[2]
        mm = numpy.transpose(data)[3]
        ss = numpy.transpose(data)[4]
        ms = numpy.transpose(data)[5]

        self.time_stamp = hh*3600 + mm*60 + ss + ms*0.001

        #load raw data axis
        x = numpy.transpose(data)[6]
        y = numpy.transpose(data)[7]
        z = numpy.transpose(data)[8]
    
        #normalise
        x_filtered = (x  - numpy.mean(x))/numpy.std(x)
        y_filtered = (y  - numpy.mean(y))/numpy.std(y)
        z_filtered = (z  - numpy.mean(z))/numpy.std(z)

        self.stream_data = torch.zeros((4, self.items_count))

        self.stream_data[0] = torch.from_numpy(x_filtered)
        self.stream_data[1] = torch.from_numpy(y_filtered)
        self.stream_data[2] = torch.from_numpy(z_filtered)
        

    def _load_stats(self, folder):
        file_name = folder + "/anotacie_final.csv"

        self.stats = None

        if path.isfile(file_name):
            annotation_data = numpy.genfromtxt(file_name, delimiter=';')

            self.stats = numpy.zeros(self.classes_count)

            for i in range(len(annotation_data)-1):
                annotation  = int(annotation_data[i+1][0])
                class_id = self._get_class_id(annotation)
                self.stats[class_id]+= 1

    
    def _get_class_id(self, raw_id):
        class_dict = {}

        class_dict[0]   = 1
        class_dict[1]   = 1
        class_dict[2]   = 2
        class_dict[3]   = 3
        class_dict[4]   = 3
        class_dict[5]   = 0
        class_dict[6]   = 0
        class_dict[7]   = 0
        class_dict[-1]  = 0
        

        return class_dict[raw_id]

if __name__ == "__main__":
    stream = StreamMagnetometer("/Users/michal/dataset/car_detection_2/Meranie_20_06_03-Kinekus/1", 512)

    ts, data = stream.get_window(100)

    print(ts)
    print(data)