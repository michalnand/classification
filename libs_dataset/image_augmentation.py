import numpy
from PIL import Image, ImageEnhance


class ImageAugmentation:

    def __init__(self):
        self.rotation_max       = 30.0
        
        self.brightness_min     = 0.5
        self.brightness_max     = 1.5

        self.contrast_min       = 0.0
        self.contrast_max       = 2.0

        self.color_min          = 0.5
        self.color_max          = 1.5

    def process(self, image_input):
        image_pil = Image.fromarray(image_input)   
        
        image_pil = self._random_rotation(image_pil)
        image_pil = self._random_colors(image_pil)
       
        image_np  = numpy.array(image_pil)/255.0
        return image_np

    def _random_rotation(self, image_input):
        angle = self._rnd(-self.rotation_max, self.rotation_max)
        return image_input.rotate(angle)
  
    def _random_colors(self, image_input):
        image_result    = image_input

        br              = self._rnd(self.brightness_min, self.brightness_max)
        con             = self._rnd(self.contrast_min, self.contrast_max)
        col             = self._rnd(self.color_min, self.color_max)
        
        fil             = ImageEnhance.Brightness(image_result)
        image_result    = fil.enhance(br)

        fil             = ImageEnhance.Contrast(image_result)
        image_result    = fil.enhance(con)

        fil             = ImageEnhance.Color(image_result)
        image_result    = fil.enhance(col)

        return image_result

    def _rnd(self, min_value, max_value):
        return (max_value - min_value)*numpy.random.rand() + min_value

if __name__ == "__main__":
    input_image = Image.open("02.jpg")

    augm = ImageAugmentation()

    result = augm.process(numpy.array(input_image))

    print("\n\n\n\n")
    print(">>> ", result.shape)
    print("\n\n\n\n")

    output_image = Image.fromarray(numpy.array(result*255, dtype=numpy.uint8) )

    output_image.show()