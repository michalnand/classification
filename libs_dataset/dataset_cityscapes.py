import numpy
import torch
import os

from .images_loader import *

from PIL import Image

class DatasetCityscapes:

    def __init__(self, root_path, height = 256, width = 512):

        training_cities = []
        training_cities.append("aachen")
        training_cities.append("bochum")
        training_cities.append("bremen")
        training_cities.append("cologne")
        training_cities.append("darmstadt")
        training_cities.append("dusseldorf")
        training_cities.append("erfurt")
        training_cities.append("hamburg")
        training_cities.append("hanover")
        training_cities.append("jena")
        training_cities.append("krefeld")
        training_cities.append("monchengladbach")
        training_cities.append("strasbourg")
        training_cities.append("stuttgart")
        training_cities.append("tubingen")
        training_cities.append("ulm")
        training_cities.append("weimar")
        training_cities.append("zurich")

        self.training_images    = []
        self.training_masks     = []
        

        self.training_count = 0

        classes_count = 0

        for city in training_cities:
            images = ImagesLoader([root_path + "leftImg8bit_trainvaltest/leftImg8bit/" + "train/" + city + "/"], height, width, channel_first=True)
            masks  = ImagesLoader([root_path + "gtFine_trainvaltest/gtFine/" + "train/" + city + "/"], height, width, channel_first=True, file_mask = "labelIds")
            
            self.training_images.append(images)
            self.training_masks.append(masks)
            
            self.training_count+= images.count

            cc = masks.images.max()
            if cc > classes_count:
                classes_count = cc



        testing_cities  = []
        testing_cities.append("berlin")
        testing_cities.append("bielefeld")
        testing_cities.append("bonn")
        testing_cities.append("leverkusen")
        testing_cities.append("mainz")
        testing_cities.append("munich")
       


        self.testing_images    = []
        self.testing_masks     = []
        
        self.testing_count = 0

        for city in testing_cities:
            images = ImagesLoader([root_path + "leftImg8bit_trainvaltest/leftImg8bit/" + "test/" + city + "/"], height, width, channel_first=True)
            masks  = ImagesLoader([root_path + "gtFine_trainvaltest/gtFine/" + "test/" + city + "/"], height, width, channel_first=True, file_mask = "labelIds")
            
            self.testing_images.append(images)
            self.testing_masks.append(masks)
            
            self.testing_count+= images.count

            cc = masks.images.max()
            if cc > classes_count:
                classes_count = cc


        '''
        for i in range(4):
            city_idx  = numpy.random.randint(len(self.testing_images))
            image_idx = numpy.random.randint(self.testing_images[city_idx].count)

            image_np = self.testing_images[city_idx].images[image_idx]
            mask_np  = self.testing_masks[city_idx].images[image_idx]

            im = Image.fromarray(image_np)
            im.show()

            im = Image.fromarray(mask_np)
            im.show()
        '''


        self.channels      = 3
        self.height        = height
        self.width         = width
        self.input_shape   = (self.channels, self.height, self.width)

        self.classes_count = classes_count + 1 #36 classes
        self.output_shape  = (self.classes_count, self.height, self.width)

        memory = (self.get_training_count() + self.get_testing_count())*2*numpy.prod(self.input_shape)

        print("\n\n\n\n")
        print("dataset summary : \n")
        print("training_count = ", self.get_training_count())
        print("testing_count  = ", self.get_testing_count())
        print("channels = ", self.channels)
        print("height   = ", self.height)
        print("width    = ", self.width)
        print("classes_count =  ", self.classes_count)
        print("required_memory = ",  memory/1000000, " MB")
        print("\n")

    
    def get_training_count(self):
        return self.training_count

    def get_testing_count(self):
        return self.testing_count

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_images, self.training_masks, batch_size)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.training_images, self.training_masks, batch_size)

    def _get_batch(self, images, masks, batch_size):
        result_x = torch.zeros((batch_size, self.channels, self.height, self.width)).float()
        result_y = torch.zeros((batch_size, self.classes_count, self.height, self.width)).float()

        for i in range(batch_size): 
            city_idx  = numpy.random.randint(len(images))
            image_idx = numpy.random.randint(images[city_idx].count)

            image     = numpy.array(images[city_idx].images[image_idx])/256.0
            mask      = numpy.array(masks[city_idx].images[image_idx]).mean(axis=0).astype(int)

            mask_one_hot = numpy.eye(self.classes_count)[mask]
            mask_one_hot = numpy.moveaxis(mask_one_hot, 2, 0)

            result_x[i]  = torch.from_numpy(image).float()
            result_y[i]  = torch.from_numpy(mask_one_hot).float()

        return result_x, result_y

       


if __name__ == "__main__":
    dataset = DatasetCityscapes("/Users/michal/dataset/cityscapes/")


    x, y = dataset.get_training_batch(batch_size=64)

    print(x.shape, y.shape)