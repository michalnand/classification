import numpy
import models.model_1.model as Model
from PIL import Image

from segmentation_inference import *


si = SegmentationInference(Model, "models/model_1/trained/", 5)


def compute(si, input_file_name_prefix):
    image       = Image.open(input_file_name_prefix + ".jpg")
    image       = image.resize((512, 256))
    image_np    = numpy.array(image)

    prediction_np, mask, result = si.process(image_np)


    image = Image.fromarray((result*255).astype(numpy.uint8), 'RGB')
    image.save(input_file_name_prefix + "_masked.jpg")

    image = Image.fromarray((mask*255).astype(numpy.uint8), 'RGB')
    image.save(input_file_name_prefix + "_mask.jpg")
    

compute(si, "images/0")
compute(si, "images/1")
compute(si, "images/2")
compute(si, "images/3")
compute(si, "images/4")
compute(si, "images/5")
compute(si, "images/6")
compute(si, "images/7")
