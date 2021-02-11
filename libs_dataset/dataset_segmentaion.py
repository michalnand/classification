import numpy
import torch 
import os

from images_loader import *

from PIL import Image, ImageEnhance

class DatasetSegmentation:

    def __init__(self, folders_training, folders_testing, classes_ids, height = 480, width = 640):

        self.classes_ids        = classes_ids
        self.classes_count      = len(classes_ids)
        self.height             = height
        self.width              = width

        self.training_images    = []
        self.training_masks     = []
        self.training_count     = 0

        for folder in folders_training:
            images = ImagesLoader([folder + "/images/"], height, width, channel_first=True)
            masks  = ImagesLoader([folder + "/mask/"], height, width, channel_first=True, file_mask = "_watershed_mask", postprocessing = self._mask_postprocessing)
            
            self.training_images.append(images)
            self.training_masks.append(masks)
            
            self.training_count+= images.count


        self.testing_images    = []
        self.testing_masks     = []
        self.testing_count = 0

        for folder in folders_testing:
            images = ImagesLoader([folder + "/images/"], height, width, channel_first=True)
            masks  = ImagesLoader([folder + "/mask/"], height, width, channel_first=True, file_mask = "_watershed_mask", postprocessing = self._mask_postprocessing)
            
            self.testing_images.append(images)
            self.testing_masks.append(masks)
            
            self.testing_count+= images.count


        self.channels      = 3
        self.height        = height
        self.width         = width
        self.input_shape   = (self.channels, self.height, self.width)

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
            group_idx = numpy.random.randint(len(images))
            image_idx = numpy.random.randint(images[group_idx].count)

            image_np    = numpy.array(images[group_idx].images[image_idx])/256.0

            mask_np     = numpy.array(masks[group_idx].images[image_idx]).mean(axis=0).astype(int)
            
            if augmentation:
                image_np  = self._augmentation_noise(image_np)
                image_np, mask_np  = self._augmentation_flip(image_np, mask_np)

            mask_one_hot = numpy.eye(self.classes_count)[mask_np]
            mask_one_hot = numpy.moveaxis(mask_one_hot, 2, 0)

            result_x[i]  = torch.from_numpy(image_np).float()
            result_y[i]  = torch.from_numpy(mask_one_hot).float()

        return result_x, result_y


    def _augmentation_noise(self, image_np):
        brightness = self._rnd(-0.25, 0.25)
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


    def _mask_postprocessing(self, image):
        image    = image.resize((self.width, self.height), Image.NEAREST)
        image    = image.convert("L")
        
        for i in range(len(self.classes_ids)):
            image.putpixel((10*i, 10*i), self.classes_ids[i])

        image = image.quantize(self.classes_count)

        return image
       


if __name__ == "__main__":

    folders_training = []
    folders_training.append("/Users/michal/dataset/outdoor/lietavska_lucka/")
    folders_training.append("/Users/michal/dataset/outdoor/istrobotics_0/")
    folders_training.append("/Users/michal/dataset/outdoor/istrobotics_1/")


    classes_ids     = [8, 12, 21, 22, 23]

    
    dataset = DatasetSegmentation(folders_training, folders_training, classes_ids)

    batch_size = 4

    x, y = dataset.get_testing_batch(batch_size)
    x, y = dataset.get_training_batch(batch_size)


    classes_count   = len(classes_ids)
    
    for i in range(batch_size):

        image_np = x[i].detach().to("cpu").numpy()
        mask_np  = y[i].detach().to("cpu").numpy()


        image_np    = (255.0*image_np).astype(numpy.uint8)

        tmp         = numpy.zeros((mask_np.shape[1], mask_np.shape[2]))

        for ch in range(classes_count):
            tmp+= (ch/(classes_count-1))*mask_np[ch]

        mask_np     = (255.0*tmp).astype(numpy.uint8)

        image_np    = numpy.moveaxis(image_np, 0, 2)

        im = Image.fromarray(image_np, 'RGB')
        im.show()

        im = Image.fromarray(mask_np)
        im.show()