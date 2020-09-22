import numpy
from os import walk
from PIL import Image
import torch


class BackgroundImageLoader:
    def __init__(self, folders_paths, width = 96, height = 96, count_per_image = 20, grayscale = False):

        self.width  = width
        self.height = height
        self.count_per_image = count_per_image
        self.grayscale = grayscale

        if self.grayscale:
            self.channels = 1
        else:
            self.channels = 3

        self.file_names = []
        for folder in folders_paths:
            self.file_names = self.file_names + self._find_files(folder)

        self.file_names.sort()


        self.count = self.count_per_image*len(self.file_names)

        print("count = ", self.count)
        print("channels     = ", self.channels)
        print("height       = ", self.height)
        print("width        = ", self.width)

        self.images = numpy.zeros((self.count, self.channels, self.height, self.width))
 
        ptr = 0
        for file_name in self.file_names:
            print("processing image :", file_name)
            result = self._process_image(file_name)
            for i in range(self.count_per_image):
                self.images[ptr] = result[i]
                ptr+= 1

    def get_batch(self, batch_size = 32):
        result = torch.zeros((batch_size, self.channels, self.height, self.width))

        for i in range(batch_size):
            idx = numpy.random.randint(self.count)
            result[i] = torch.from_numpy(self.images[idx])

        return result

    def show(self, idx):
        if self.channels == 3:
            image_np = 0.3*self.images[idx][0] + 0.59*self.images[idx][1] + 0.11*self.images[idx][1]
        else:
            image_np = self.images[idx][0]

        image = Image.fromarray(image_np*255)
        image.show()

    def _find_files(self, path):
        files = []
        for (dirpath, dirnames, filenames) in walk(path):
            files.append(filenames)

        result = []
        for file_name in files[0]:
            result.append(path + file_name)

        return result

    def _process_image(self, file_name):
        result = numpy.zeros((self.count_per_image, self.channels, self.height, self.width))

        image = Image.open(file_name)

        for i in range(self.count_per_image):
            x_ofs = numpy.random.randint(image.width - self.width - 1)
            y_ofs = numpy.random.randint(image.height - self.height - 1)

            crop_area = (x_ofs, y_ofs, x_ofs + self.width, y_ofs + self.height)
            cropped_img = image.crop(crop_area)

            image_np = numpy.array(cropped_img)
            image_np = numpy.rollaxis(image_np, 2, 0) 

            if self.grayscale or True:
                image_np = 0.3*image_np[0] + 0.59*image_np[1] + 0.11*image_np[2] 
                result[i][0] = numpy.clip(image_np/256.0, 0.0, 1.0)
            else:
                result[i][0] = numpy.clip(image_np/256.0, 0.0, 1.0) 

        return result



if __name__ == "__main__":
    dirs = []
    
    dirs.append("/Users/michal/dataset/background/city/")
    dirs.append("/Users/michal/dataset/background/indoor/")
    dirs.append("/Users/michal/dataset/background/nature/")
    dirs.append("/Users/michal/dataset/background/paper/")

    background = BackgroundImageLoader(dirs, grayscale=True)

    background.show(100)
    background.show(150)
    background.show(200)
    background.show(250)

    print("done")
    