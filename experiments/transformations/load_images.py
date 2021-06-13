import numpy
from PIL import Image, ImageFilter


class LoadImageTransformed:
    def __init__(self, file_name, height = 256, width = 256, source_height = 960, source_width = 1280, augmentations_count = 4):
        self.height             = height
        self.width              = width 
        self.source_height      = source_height
        self.source_width       = source_width 

        image       = Image.open(file_name).convert("RGB")
        image       = image.resize((source_width, source_height))

        images_aug, target = self._make_transformations(image, augmentations_count)
          
        self.images = images_aug/256.0
        self.target = target


    def _make_transformations(self, image_in, augmentation_count):
        images_result           = numpy.zeros((augmentation_count, 2, 3, self.height, self.width), dtype=numpy.uint8)
        transformations_result  = numpy.zeros((augmentation_count, 4))

        for i in range(augmentation_count):
            image_a, image_b, transformation = self._make_transformation(image_in)

            images_result[i][0]       = image_a
            images_result[i][1]       = image_b 
            transformations_result[i] = transformation

        return images_result, transformations_result


    def _make_transformation(self, image_in):
        max_distance = 0.5
        ax = numpy.random.randint(0, self.source_width - self.width//2)
        ay = numpy.random.randint(0, self.source_height - self.height//2)
        image_a, angle_a, zoom_a = self._create_transformed(image_in, ax, ay)

        if numpy.random.rand() < 0.5: 
            bx = int(ax + self._rnd(-max_distance*self.width, max_distance*self.width))
            by = int(ay + self._rnd(-max_distance*self.height, max_distance*self.height))
        else: 
            bx = int(ax + self._rnd(-max_distance*0.1*self.width, max_distance*0.1*self.width))
            by = int(ay + self._rnd(-max_distance*0.1*self.height, max_distance*0.1*self.height))
        
        image_b, angle_b, zoom_b = self._create_transformed(image_in, bx, by)
            
        transformation      = numpy.zeros(4, )
        transformation[0]   = (ax - bx)/self.width
        transformation[1]   = (ay - by)/self.height
        transformation[2]   = (angle_a - angle_b)*numpy.pi/180.0
        transformation[3]   = zoom_a/zoom_b
 
        return image_a, image_b, transformation


    def _create_transformed(self, image_in, x, y):
        
        angle   = 0 #self._rnd(-10.0, 10.0)
        zoom    = 1 #self._rnd(0.9, 1.1)   

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

        image_res   = numpy.array(image_res)  
        image_res   = numpy.moveaxis(image_res, 2, 0)
        
        return image_res, angle, zoom

    def _rnd(self, min_value, max_value):
        return (max_value - min_value)*numpy.random.rand() + min_value

