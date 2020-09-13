import numpy
import torch
from PIL import Image, ImageDraw, ImageFont


class DatasetLineFollower:
    def __init__(self, width = 8, height = 8, classes_count = 5, training_count = 50000, testing_count = 1000):
        self.angle_noise      = 0.05
        self.top_bottom_noise = 0.1 

        self.noise_level      = 0.05

        self.line_width_min = 0.15
        self.line_width_max = 0.2

        self.intensity_min = 0.5
        self.intensity_max = 1.0

        

        self.width         = width
        self.height        = height
        self.input_shape   = (1, self.height, self.width)

        self.classes_count = classes_count
        self.output_shape  = (self.classes_count, )


        self.training_count = training_count
        self.testing_count  = testing_count

        self.training_x = numpy.zeros((self.training_count, ) + self.input_shape, dtype=numpy.float32)
        self.training_y = numpy.zeros((self.training_count, ) + self.output_shape, dtype=numpy.float32)
        self.testing_x  = numpy.zeros((self.testing_count, ) + self.input_shape, dtype=numpy.float32)
        self.testing_y  = numpy.zeros((self.testing_count, ) + self.output_shape, dtype=numpy.float32)


        for i in range(self.training_count):
            training_x, training_y = self._create_item()
            self.training_x[i] = training_x.copy()
            self.training_y[i] = training_y.copy()


        for i in range(self.testing_count):
            testing_x, testing_y = self._create_item()
            self.testing_x[i] = testing_x.copy()
            self.testing_y[i] = testing_y.copy()

        print("training_count = ", self.get_training_count())
        print("testing_count = " , self.get_testing_count())


    def get_training_count(self):
        return len(self.training_x)

    def get_testing_count(self):
        return len(self.testing_x)

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_x, self.training_y, batch_size)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_x, self.testing_y, batch_size)


    def _get_batch(self, x, y, batch_size):
        result_x = torch.zeros((batch_size, ) + self.input_shape)
        result_y = torch.zeros((batch_size, ) + self.output_shape)

        for i in range(batch_size): 
            idx = numpy.random.randint(len(x))

            result_x[i]  = torch.from_numpy(x[idx]).float()
            result_y[i]  = torch.from_numpy(y[idx]).float()

        return result_x, result_y




    def _create_item(self):
        line_position = numpy.random.rand()

        
        if numpy.random.randint(2) == 0:
            angle = 0.0
        else:
            angle = self.angle_noise*(numpy.random.rand() - 0.5)*2.0

        x0 = numpy.clip(line_position + angle, 0, 1)
        y0 = numpy.clip(0.0 + numpy.random.rand()*self.top_bottom_noise, 0, 1)
        x1 = numpy.clip(line_position - angle, 0, 1)
        y1 = numpy.clip(1.0 - numpy.random.rand()*self.top_bottom_noise, 0, 1)

        lines = [[x0, y0, x1, y1]]
          
        img     = self._generate_line_image(lines)
        img_np  =  numpy.array(img)/255.0 

        img_np  = self.noise_level*numpy.random.randn(1, self.height, self.width) + (1.0 - self.noise_level)*img_np

        img_np  = (img_np - img_np.min())/(img_np.max() - img_np.min())

        #random swap colors
        if numpy.random.randint(2) == 0:
            img_np = -img_np

        #quantize target
        class_idx = int(numpy.clip(self.classes_count*line_position, 0, self.classes_count-1))
        
        target_np = numpy.zeros(self.output_shape)
        target_np[class_idx] = 1.0

        return img_np, target_np


    def _generate_line_image(self, lines):
        img = Image.new("L", (self.width, self.height)) 

      
        line_width = int(self.width*self._rnd(self.line_width_min, self.line_width_max))

        img1 = ImageDraw.Draw(img)   

        for line in lines:
            intensity = int(self._rnd(self.intensity_min, self.intensity_max)*255)
            shape = [(line[0]*self.width, line[1]*self.height), (line[2]*self.width, line[3]*self.height)] 
            img1.line(shape, fill = intensity, width = line_width) 

        if numpy.random.randint(2) == 0:
            #random perspective
            param = numpy.random.randint(20)
            coeff = self._create_coeff(
                                        (0, 0),
                                        (img.width, 0),
                                        (img.width, img.height),
                                        (0, img.height),
                                        (-param, 0),
                                        (img.width + param, 0),
                                        (img.width, img.height),
                                        (0, img.height)
                                    )

            img = img.transform( (img.width, img.height), method=Image.PERSPECTIVE, data=coeff)
        

        return img


    def _rnd(self, min, max):
        return numpy.random.rand()*(max - min) + min

    def _create_coeff(  self,
                        xyA1, xyA2, xyA3, xyA4,
                        xyB1, xyB2, xyB3, xyB4):

        A = numpy.array([
                [xyA1[0], xyA1[1], 1, 0, 0, 0, -xyB1[0] * xyA1[0], -xyB1[0] * xyA1[1]],
                [0, 0, 0, xyA1[0], xyA1[1], 1, -xyB1[1] * xyA1[0], -xyB1[1] * xyA1[1]],
                [xyA2[0], xyA2[1], 1, 0, 0, 0, -xyB2[0] * xyA2[0], -xyB2[0] * xyA2[1]],
                [0, 0, 0, xyA2[0], xyA2[1], 1, -xyB2[1] * xyA2[0], -xyB2[1] * xyA2[1]],
                [xyA3[0], xyA3[1], 1, 0, 0, 0, -xyB3[0] * xyA3[0], -xyB3[0] * xyA3[1]],
                [0, 0, 0, xyA3[0], xyA3[1], 1, -xyB3[1] * xyA3[0], -xyB3[1] * xyA3[1]],
                [xyA4[0], xyA4[1], 1, 0, 0, 0, -xyB4[0] * xyA4[0], -xyB4[0] * xyA4[1]],
                [0, 0, 0, xyA4[0], xyA4[1], 1, -xyB4[1] * xyA4[0], -xyB4[1] * xyA4[1]],
                ], dtype=numpy.float32)

        B = numpy.array([
                xyB1[0],
                xyB1[1],
                xyB2[0],
                xyB2[1],
                xyB3[0],
                xyB3[1],
                xyB4[0],
                xyB4[1],
                ], dtype=numpy.float32)

        return numpy.linalg.solve(A, B)

if __name__ == "__main__":

    width   = 64
    height  = 64
    dataset = DatasetLineFollower(width = width, height = height, classes_count = 5, training_count = 100, testing_count = 100)

    imgs_x  = 8
    imgs_y  = 8
    spacing = 4

    result_img_height = imgs_y*(height + spacing)
    result_img_width = imgs_y*(width + spacing)
    result_img = numpy.zeros((result_img_height, result_img_width))
    x_t, y_t = dataset.get_training_batch(batch_size=imgs_x*imgs_y)


    for y in range(imgs_y):
        for x in range(imgs_x):
            idx  = y*imgs_x + x

            x_np = x_t[idx][0].to("cpu").detach().numpy()
            x_np = (x_np - x_np.min())/(x_np.max() - x_np.min())

            y_ofs = y*(height + spacing)
            x_ofs = x*(height + spacing)

            result_img[0 + y_ofs:height + y_ofs, 0 + x_ofs:width + x_ofs] = x_np

    
    
    img_gr  = Image.fromarray(255*result_img)
    img     = Image.new('RGB', img_gr.size)
    img.paste(img_gr)

    target_height  = 96
    target_width   = 96

    height_ = target_height*imgs_y
    width_  = target_width*imgs_y

    img = img.resize((height_, width_), Image.BICUBIC)

    y_np = y_t.to("cpu").detach().numpy()

    for y in range(imgs_y):
        for x in range(imgs_x):
            idx  = y*imgs_x + x

            text = str(numpy.argmax(y_np[idx]))

            font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 25)

            y_ofs = y*(target_height + spacing)
            x_ofs = x*(target_width  + spacing)

            draw = ImageDraw.Draw(img) 
            draw.text((x_ofs, y_ofs), text, (200, 0, 0),font=font)

    img.show()  
    