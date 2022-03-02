import numpy
import torch
from PIL import Image
import load_images

import models.model_0.model as Model


def show_image(image_np):
    image   = (255.0*image_np).astype(numpy.uint8)    
    image   = numpy.moveaxis(image, 0, 2)
    im      = Image.fromarray(image, 'RGB')
    im.show()

loader = load_images.LoadImageTransformed("images/image_2.jpg", 256, 256, 480, 640, augmentations_count = 8)

x_input = loader.images
y_target = loader.target

show_image(x_input[0][0])
show_image(x_input[0][1])



model = Model.Create((2, 3, 256, 256), (4, ))
model.load("./models/model_0/trained/")

batch = torch.from_numpy(x_input).float().to(model.device)
 

y_predicted = model(batch).detach().to("cpu").numpy()

error = ((y_target - y_predicted)**2).mean()

print(">>> ", x_input.shape, y_predicted.shape)
print(y_target) 
print(y_predicted)
print(error)
