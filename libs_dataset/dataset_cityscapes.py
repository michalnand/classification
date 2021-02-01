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
        return self._get_batch(self.training_images, self.training_masks, batch_size, True)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.training_images, self.training_masks, batch_size, False)

    def _get_batch(self, images, masks, batch_size, augmentation = False):
        result_x = torch.zeros((batch_size, self.channels, self.height, self.width)).float()
        result_y = torch.zeros((batch_size, self.classes_count, self.height, self.width)).float()

        for i in range(batch_size): 
            city_idx  = numpy.random.randint(len(images))
            image_idx = numpy.random.randint(images[city_idx].count)

            image_np    = numpy.array(images[city_idx].images[image_idx])/256.0

            mask_np     = numpy.array(masks[city_idx].images[image_idx]).mean(axis=0).astype(int)
            
            if augmentation:
                image_np  = self._augmentation_noise(image_np)
                image_np, mask_np  = self._augmentation_flip(image_np, mask_np)

            mask_one_hot = numpy.eye(self.classes_count)[mask_np]
            mask_one_hot = numpy.moveaxis(mask_one_hot, 2, 0)

            result_x[i]  = torch.from_numpy(image_np).float()
            result_y[i]  = torch.from_numpy(mask_one_hot).float()

        return result_x, result_y


    def _augmentation_noise(self, image_np):
        brightness = self._rnd(-0.5, 0.5)
        contrast   = self._rnd(0.5, 1.5)
        noise      = 0.05*(2.0*numpy.random.rand(self.channels, self.height, self.width) - 1.0)

        result     = image_np + brightness
        result     = 0.5 + contrast*(result - 0.5)
        result     = result + noise

        result     = numpy.clip(result, 0.0, 1.0)

        return result

    def _augmentation_flip(self, image_np, mask_np, p = 0.2):
        #random flips
        if self._rnd(0, 1) < p:
            image_np    = numpy.flip(image_np, axis=1)
            mask_np     = numpy.flip(mask_np,  axis=0)

        if self._rnd(0, 1) < p:
            image_np    = numpy.flip(image_np, axis=2)
            mask_np     = numpy.flip(mask_np,  axis=1)

        #random rolling
        if self._rnd(0, 1) < p:
            r           = numpy.random.randint(-32, 32)
            image_np    = numpy.roll(image_np, r, axis=1)
            mask_np     = numpy.roll(mask_np, r, axis=0)

        if self._rnd(0, 1) < p:
            r           = numpy.random.randint(-32, 32)
            image_np    = numpy.roll(image_np, r, axis=2)
            mask_np     = numpy.roll(mask_np, r, axis=1)

        return image_np.copy(), mask_np.copy()

    def _rnd(self, min_value, max_value):
        return (max_value - min_value)*numpy.random.rand() + min_value

       


if __name__ == "__main__":
    dataset = DatasetCityscapes("/Users/michal/dataset/cityscapes/")

    batch_size = 16

    x, y = dataset.get_testing_batch(batch_size)
    x, y = dataset.get_training_batch(batch_size)
    
    
    print(x.shape, y.shape)

    for i in range(batch_size):

        image_np = x[i].detach().to("cpu").numpy()
        mask_np  = y[i].detach().to("cpu").numpy()

        image_np    = (255.0*image_np).astype(numpy.uint8)
        mask_np     = (255.0*mask_np[1]).astype(numpy.uint8)

        image_np    = numpy.moveaxis(image_np, 0, 2)

        im = Image.fromarray(image_np, 'RGB')
        im.show()

        im = Image.fromarray(mask_np)
        im.show()
