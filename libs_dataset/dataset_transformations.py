import numpy
import torch 
import os

from .images_loader import *
#from images_loader import *

from PIL import Image, ImageFilter

class DatasetTransformations:

    def __init__(self, folders_training, folders_testing, height = 256, width=256, source_height = 960, source_width = 1280, augmentation_count = 64):

        self.height             = height
        self.width              = width 
        self.source_height      = source_height
        self.source_width       = source_width

        self.augmentation_count = augmentation_count

        self.training_images    = []
        self.training_targets   = []
        self.training_count     = 0

        self.classes_count = 4

        for folder in folders_training:
            images = ImagesLoader([folder], source_height, source_width, channel_first=True)

            images_aug, target = self._make_transformations(images.images)
            
            self.training_images.append(images_aug)
            self.training_targets.append(target)

            self.training_count+= images_aug.shape[0]
 

        self.testing_images    = []
        self.testing_targets   = []
        self.testing_count     = 0
        for folder in folders_testing:
            images = ImagesLoader([folder], source_height, source_width, channel_first=True)

            images_aug, target = self._make_transformations(images.images)
            
            self.testing_images.append(images_aug)
            self.testing_targets.append(target)

            self.testing_count+= images_aug.shape[0]

        self.channels      = 3
        self.height        = height
        self.width         = width

        self.input_shape   = (self.channels, 2, self.height, self.width)
        self.output_shape  = (self.classes_count, )

        memory = (self.get_training_count() + self.get_testing_count())*numpy.prod(self.input_shape)

        print("\n\n\n\n")
        print("dataset summary : \n")
        print("training_count = ", self.get_training_count())
        print("testing_count  = ", self.get_testing_count())
        print("channels = ", self.channels)
        print("height   = ", self.height)
        print("width    = ", self.width)
        print("transformations_count =  ", self.classes_count)
        print("required_memory = ",  memory/1000000, " MB")
        print("\n")

    
    def get_training_count(self):
        return self.training_count

    def get_testing_count(self):
        return self.testing_count

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_images, self.training_targets, batch_size)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.training_images, self.training_targets, batch_size)

    def _get_batch(self, images, targets, batch_size):
        result_x = torch.zeros((batch_size, 2, self.channels, self.height, self.width)).float()
        result_y = torch.zeros((batch_size, self.classes_count)).float()

        for i in range(batch_size): 
            group_idx = numpy.random.randint(len(images))
            image_idx = numpy.random.randint(len(images[group_idx]))

            image_np    = numpy.array(images[group_idx][image_idx])/256.0
  
            target_np   = numpy.array(targets[group_idx][image_idx])
            
            result_x[i]  = torch.from_numpy(image_np).float()
            result_y[i]  = torch.from_numpy(target_np).float()

        return result_x, result_y

    def _make_transformations(self, images):
        
        count           = images.shape[0]
        total_count     = count*self.augmentation_count

        images_result           = numpy.zeros((total_count, 2, 3, self.height, self.width), dtype=numpy.uint8)
        transformations_result  = numpy.zeros((total_count, self.classes_count))

        ptr = 0
        for j in range(count):
            image_in = Image.fromarray(numpy.moveaxis(images[j], 0, 2), 'RGB')

            for i in range(self.augmentation_count):
                image_a, image_b, transformation = self._make_transformation(image_in)

                images_result[ptr][0]       = image_a
                images_result[ptr][1]       = image_b
                transformations_result[ptr] = transformation
                ptr+= 1

        return images_result, transformations_result


    def _make_transformation(self, image_in):
        max_distance = 0.5
        ax = numpy.random.randint(0, self.source_width - self.width//2)
        ay = numpy.random.randint(0, self.source_height - self.height//2)
        image_a, angle_a, zoom_a = self._create_transformed(image_in, ax, ay)
   
        if True: #numpy.random.rand() < 0.5: 
            bx = int(ax + self._rnd(-max_distance*self.width, max_distance*self.width))
            by = int(ay + self._rnd(-max_distance*self.height, max_distance*self.height))
            image_b, angle_b, zoom_b = self._create_transformed(image_in, bx, by)
            close = 1.0
        else:
            dx = self._rnd(self.width//2, self.source_width - self.width)
            dy = self._rnd(self.height//2, self.source_height - self.height)
            
            if numpy.random.rand() < 0.5:
                dx = -dx

            if numpy.random.rand() < 0.5:
                dy = -dy

            bx = int(ax + dx) 
            by = int(ay + dy)

            image_b, angle_b, zoom_b = self._create_transformed(image_in, bx, by)
            close = 0.0

        transformation      = numpy.zeros(self.classes_count, )
        transformation[0]   = (ax - bx)/self.width
        transformation[1]   = (ay - by)/self.height
        transformation[2]   = (angle_a - angle_b)*numpy.pi/180.0
        transformation[3]   = zoom_a/zoom_b

 
        return image_a, image_b, transformation


    def _create_transformed(self, image_in, x, y):
        
        angle   = self._rnd(-10.0, 10.0)
        zoom    = self._rnd(0.9, 1.1)   

        ax      = int(x - 0.5*zoom*self.width)
        ay      = int(y - 0.5*zoom*self.height)
        bx      = int(x + 0.5*zoom*self.width)
        by      = int(y + 0.5*zoom*self.height)

        if ax < 0: 
            ax = 0

        if bx > self.source_width-1:
            bx = self.source_width-1

        if ay < 0:
            ay = 0

        if by > self.source_height-1:
            by = self.source_height-1

        #random rotation
        image_res = image_in.rotate(angle)

        #random zoom
        image_res = image_res.crop((ax, ay, bx, by))

        #fit to scale
        image_res = image_res.resize((self.width, self.height))

        #apply random filter
        if numpy.random.rand() < 0.5:
            fil = numpy.random.randint(6)

            if fil == 0:
                image_res   = image_res.filter(ImageFilter.BLUR)
            elif fil == 1:
                image_res   = image_res.filter(ImageFilter.EDGE_ENHANCE)
            elif fil == 2:
                image_res   = image_res.filter(ImageFilter.EDGE_ENHANCE_MORE)
            elif fil == 3:
                image_res   = image_res.filter(ImageFilter.SHARPEN)
            elif fil == 4:
                image_res   = image_res.filter(ImageFilter.SMOOTH)
            elif fil == 5:
                image_res   = image_res.filter(ImageFilter.SMOOTH_MORE)

        image_res   = numpy.array(image_res)  
        image_res   = numpy.moveaxis(image_res, 2, 0)
        
        return image_res, angle, zoom

    def _rnd(self, min_value, max_value):
        return (max_value - min_value)*numpy.random.rand() + min_value




if __name__ == "__main__":

    folders_training = []
    folders_training.append("/Users/michal/dataset/outdoor/lietavska_lucka/images/")
    #folders_training.append("/Users/michal/dataset/outdoor/istrobotics_0/")
    #folders_training.append("/Users/michal/dataset/outdoor/street/")

    
    dataset = DatasetTransformations(folders_training, folders_training)

    batch_size = 16


    x, y = dataset.get_training_batch(batch_size)
    #x, y = dataset.get_testing_batch(batch_size)
    
    for i in range(batch_size):
        image_np    = x[i].detach().to("cpu").numpy()
        image_np    = (255.0*image_np).astype(numpy.uint8)
        
        image_a    = numpy.moveaxis(image_np[0], 0, 2)
        im = Image.fromarray(image_a, 'RGB')
        im.show()

        image_b    = numpy.moveaxis(image_np[1], 0, 2)
        im = Image.fromarray(image_b, 'RGB')
        im.show()


        label       = y[i].detach().to("cpu").numpy()

        print(">>> ", i, " : ", label)