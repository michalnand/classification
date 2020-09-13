import numpy
from PIL import Image


class ImageAugmentation:

    def __init__(self):
        self.resize_min     = 0.25
        self.resize_max     = 2.0
        self.rotation_max   = 30.0
        self.target_width   = 512
        self.target_height  = 512
        self.offset_noise   = 0.5
        self.white_noise    = 0.2


    def process(self, image_input):
        image_pil = Image.from_array(image_input)   
        
        image_pil = self._random_flip(image_pil)
        image_pil = self._random_resize(image_pil)
        image_pil = self._random_rotation(image_pil)
        image_pil = self._random_crop(image_pil)
        image_pil = self._random_colors(image_pil)
        image_pil = self._random_offset_noise(image_pil)
        image_pil = self._random_white_noise(image_pil)

        image_np  = numpy.array(image_pil)
        image_np  = (image_np - image_np.mean())/image_np.std()

        return image_np

    def _random_flip(self, image_input):
        return image_input

    def _random_resize(self, image_input):
        #self.resize_min     = 0.5
        #self.resize_max     = 2.0
        return image_input

    def _random_rotation(self, image_input):
        #self.rotation_max
        return image_input

    def _random_crop(self, image_input):
        #self.rotation_max
        return image_input

    def _random_colors(self, image_input):
        #self.offset_noise
        return image_input

    def _random_offset_noise(self, image_input):
        #self.offset_noise
        return image_input

    def _random_white_noise(self, image_input):
        #self.white_noise
        return image_input