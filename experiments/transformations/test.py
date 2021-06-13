import numpy
from PIL import Image
import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset
import torch

import models.model_0.model as Model0

def rnd(min_value, max_value):
    return (max_value - min_value)*numpy.random.rand() + min_value

def create_transformed(image_in, x, y, height, width, source_height, source_width):
    angle   = rnd(-10.0, 10.0)
    zoom    = rnd(0.9, 1.1)   
 
    ax      = int(x - 0.5*zoom*width)
    ay      = int(y - 0.5*zoom*height)
    bx      = int(x + 0.5*zoom*width)
    by      = int(y + 0.5*zoom*height)

    if ax < 0: 
        ax = 0

    if bx > source_width-1:
        bx = source_width-1

    if ay < 0:
        ay = 0

    if by > source_height-1:
        by = source_height-1

    #random rotation
    image_res = image_in.rotate(angle)

    #random zoom
    image_res = image_res.crop((ax, ay, bx, by))

    #fit to scale
    image_res = image_res.resize((width, height))

    image_res   = numpy.array(image_res)  
    image_res   = numpy.moveaxis(image_res, 2, 0)/255.0
    
    return image_res, angle, zoom 

def make_transformation(file_name, height = 256, width=256, source_height = 960, source_width = 1280):
    image       = Image.open(file_name).convert("RGB")
    image       = image.resize((source_width, source_height))

    max_distance = 0.5
    ax = numpy.random.randint(0, source_width - width//2)
    ay = numpy.random.randint(0, source_height - height//2)
    image_a, angle_a, zoom_a = create_transformed(image, ax, ay, height, width, source_height, source_width)

    if numpy.random.rand() < 0.5: 
        bx = int(ax + rnd(-max_distance*width, max_distance*width))
        by = int(ay + rnd(-max_distance*height, max_distance*height))
        image_b, angle_b, zoom_b = create_transformed(image, bx, by, height, width, source_height, source_width)
    else: 
        bx = int(ax + rnd(-max_distance*0.1*width, max_distance*0.1*width))
        by = int(ay + rnd(-max_distance*0.1*height, max_distance*0.1*height))
        image_b, angle_b, zoom_b = create_transformed(image, bx, by, height, width, source_height, source_width)


    transformation      = numpy.zeros(4, )
    transformation[0]   = (ax - bx)/width
    transformation[1]   = (ay - by)/height
    transformation[2]   = (angle_a - angle_b)*numpy.pi/180.0
    transformation[3]   = zoom_a/zoom_b

    return image_a, image_b, transformation

def show_image(image_np):
    image   = (255.0*image_np).astype(numpy.uint8)    
    image   = numpy.moveaxis(image, 0, 2)
    im      = Image.fromarray(image, 'RGB')
    im.show()

image_a, image_b, transformation = make_transformation("images/image_3.jpg")

show_image(image_a)
show_image(image_b)


batch = torch.zeros((1, 2, 3, 256, 256))

batch[0][0] = torch.from_numpy(image_a)
batch[0][1] = torch.from_numpy(image_b) 


model = Model0.Create((2, 3, 256, 256), (4, ))
model.load("./models/model_0/trained/") 


y = model(batch).detach().to("cpu").numpy()

error = ((y - transformation)**2).mean()

print(">>> ", y.shape)
print(transformation)
print(y[0])
print(error)