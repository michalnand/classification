import numpy
import models.model_0.model as Model
from PIL import Image

from segmentation_inference import *


#si = SegmentationInference(Model, "models/model_0/trained/")
si = SegmentationInference(Model, None, 5)


def compute(si, input_file_name_prefix):
    image       = Image.open(input_file_name_prefix + ".png")
    image       = image.resize((640, 480))
    image_np    = numpy.array(image)

    prediction_np, mask, result = si.process(image_np)


    image = Image.fromarray((result*255).astype(numpy.uint8), 'RGB')
    image.save(input_file_name_prefix + "_masked.jpg")

    image = Image.fromarray((mask*255).astype(numpy.uint8), 'RGB')
    image.save(input_file_name_prefix + "_mask.jpg")
    

compute(si, "images/01")
compute(si, "images/02")
compute(si, "images/03")
