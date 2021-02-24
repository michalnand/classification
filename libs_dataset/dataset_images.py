import numpy
import torch 
import os

from .images_loader import *
from .tiny_imagenet_config import *

from PIL import Image, ImageFilter

class DatasetImages:

    def __init__(self, folders_training, classes_training, folders_testing, classes_testing, height = 64, width = 64, augmentation_count = 5):

        self.classes_count      = numpy.max(classes_training) + 1

        self.height             = height
        self.width              = width

        self.training_images    = []
        self.training_classes   = []
        self.training_count     = 0

        for folder, class_id in zip(folders_training, classes_training):
            print("loading ", folder)
            images  = ImagesLoader([folder], height, width, channel_first=True)
            classes = [class_id]*len(images.images)
            
            self.training_images.append(images.images)
            self.training_classes.append(classes)

            print("processing augmentation\n")

            images_aug = self._augmentation(images.images, augmentation_count)
            classes = [class_id]*len(images_aug)

            self.training_images.append(images_aug)
            self.training_classes.append(classes)
            
            self.training_count+= images.count*(1 + augmentation_count) 


        self.testing_images     = []
        self.testing_classes    = []
        self.testing_count      = 0

        for folder, class_id in zip(folders_testing, classes_testing):
            images  = ImagesLoader([folder], height, width, channel_first=True)
            classes = [class_id]*len(images.images)
            
            self.testing_images.append(images.images)
            self.testing_classes.append(classes)

            self.testing_count+= images.count


        self.channels      = 3
        self.height        = height
        self.width         = width
        self.input_shape   = (self.channels, self.height, self.width)

        self.output_shape  = (self.classes_count, self.height, self.width)
        memory = (self.get_training_count() + self.get_testing_count())*numpy.prod(self.input_shape)

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
        return self._get_batch(self.training_images, self.training_classes, batch_size, True)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_images, self.testing_classes, batch_size, False)

    def _get_batch(self, images, classes, batch_size, augmentation = False):
        result_x = torch.zeros((batch_size, self.channels, self.height, self.width)).float()
        result_y = torch.zeros((batch_size, self.classes_count)).float()

        for i in range(batch_size): 
            group_idx = numpy.random.randint(len(images))
            image_idx = numpy.random.randint(len(images[group_idx])) 

            image_np    = numpy.array(images[group_idx][image_idx])/256.0
            
            if augmentation: 
                image_np  = self._augmentation_noise(image_np)
                image_np  = self._augmentation_flip(image_np)

            class_id = classes[group_idx][i] 

            result_x[i]  = torch.from_numpy(image_np).float()
            result_y[i][class_id]  = 1.0

        return result_x, result_y


    def _augmentation(self, images, augmentation_count):

        angle_max = 45
        crop_prop = 0.1

        count = images.shape[0]
        total_count = count*augmentation_count

        images_result   = numpy.zeros((total_count, images.shape[1], images.shape[2], images.shape[3]), dtype=numpy.uint8)

        ptr = 0
        for j in range(count):
 
            image_in = Image.fromarray(numpy.moveaxis(images[j], 0, 2), 'RGB')

            for i in range(augmentation_count):
                angle       = self._rnd(-angle_max, angle_max)

                image_aug   = image_in.rotate(angle)

                c_left      = int(self._rnd(0, crop_prop)*self.width)
                c_top       = int(self._rnd(0, crop_prop)*self.height)

                c_right     = int(self._rnd(1.0 - crop_prop, 1.0)*self.width)
                c_bottom    = int(self._rnd(1.0 - crop_prop, 1.0)*self.height)
            
                image_aug   = image_aug.crop((c_left, c_top, c_right, c_bottom))
                
                if numpy.random.rand() < 0.5:
                    fil = numpy.random.randint(6)
 
                    if fil == 0:
                        image_aug   = image_aug.filter(ImageFilter.BLUR)
                    elif fil == 1:
                        image_aug   = image_aug.filter(ImageFilter.EDGE_ENHANCE)
                    elif fil == 2:
                        image_aug   = image_aug.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    elif fil == 3:
                        image_aug   = image_aug.filter(ImageFilter.SHARPEN)
                    elif fil == 4:
                        image_aug   = image_aug.filter(ImageFilter.SMOOTH)
                    elif fil == 5:
                        image_aug   = image_aug.filter(ImageFilter.SMOOTH_MORE)


                image_aug   = image_aug.resize((self.width, self.height))

                image_aug   = numpy.array(image_aug)  

                image_aug   = numpy.moveaxis(image_aug, 2, 0)

                images_result[ptr]  = image_aug

                ptr+=1

        return images_result

    def _augmentation_noise(self, image_np):
        brightness = self._rnd(-0.25, 0.25)
        contrast   = self._rnd(0.5, 1.5)
        noise      = 0.05*(2.0*numpy.random.rand(self.channels, self.height, self.width) - 1.0)

        result     = image_np + brightness
        result     = 0.5 + contrast*(result - 0.5)
        result     = result + noise

        result     = numpy.clip(result, 0.0, 1.0)

        return result

    def _augmentation_flip(self, image_np, p = 0.2):
        #random flips
        if self._rnd(0, 1) < p:
            image_np    = numpy.flip(image_np, axis=1)

        if self._rnd(0, 1) < p:
            image_np    = numpy.flip(image_np, axis=2)
 
        return image_np.copy()

    def _rnd(self, min_value, max_value):
        return (max_value - min_value)*numpy.random.rand() + min_value




if __name__ == "__main__":
    dataset_path     = "/Users/michal/dataset/tiny_imagenet/"
    
    folders_training, classes_training, folders_testing, classes_testing = tiny_imagenet_config(dataset_path)

    dataset = DatasetImages(folders_training, classes_training, folders_testing, classes_testing, augmentation_count=1)

    batch_size = 16

    x, _ = dataset.get_testing_batch(batch_size)
    x, _ = dataset.get_training_batch(batch_size)
    
    for i in range(batch_size):

        image_np = x[i].detach().to("cpu").numpy()
        image_np    = (255.0*image_np).astype(numpy.uint8)

        image_np    = numpy.moveaxis(image_np, 0, 2)

        im = Image.fromarray(image_np, 'RGB')
        im.show()
