import numpy
import torch
import colorsys

if torch.cuda.is_available():
    from torch2trt import torch2trt

class SegmentationInference:
    def __init__(self, Model, model_pretrained_path, classes_count):
        channels            = 3
        height              = 480
        width               = 640

        print("creating model")
        self.model      = Model.Create((channels, height, width), (classes_count, height, width))

        if model_pretrained_path is not None:
            print("loading wights")
            self.model.load(model_pretrained_path)
        self.model.eval()

        if torch.cuda.is_available():
            print("converting model into torchRT")
            x = torch.ones((1, channels, height, width)).to("cuda")
            self.model = torch2trt(self.model, [x])

        self.colors     = self._make_colors(classes_count)

        print("SegmentationInference ready")


    def process(self, image_np, channel_first = False, alpha = 0.35):
        
        image_np        = image_np/256.0

        prediction_np   = self._predict(image_np, channel_first)

        #prediction_np   = (prediction_np == 4).astype(int)

        mask            = self.colors[prediction_np, :]
        result          = (1.0 - alpha)*image_np + alpha*mask

        return prediction_np, mask, result

    def _predict(self, image_np, channel_first):

        if channel_first == False:
            image_np    = numpy.moveaxis(image_np, 2, 0)

        image_t     = torch.from_numpy(image_np).unsqueeze(0).float()

        if torch.cuda.is_available():
            image_t = image_t.to("cuda")

        prediction_t    = self.model(image_t).squeeze(0)
        prediction_t    = torch.argmax(prediction_t, dim=0)
        prediction_np   = prediction_t.detach().to("cpu").numpy()

        return prediction_np

 

    def _make_colors(self, count):

        result = []
        result.append([0, 0, 1])
        result.append([1, 1, 0])
        result.append([0, 1, 0])
        result.append([0, 1, 1])
        result.append([1, 0, 0])

        return numpy.array(result)

    '''
    def _make_colors(self, count):

        result = []
        for i in range(count):  

            phi = 2.0*numpy.pi*i/count

            r = (numpy.sin(phi + 1.0*2.0*numpy.pi/3.0) + 1.0)/2.0
            g = (numpy.sin(phi + 2.0*2.0*numpy.pi/3.0) + 1.0)/2.0
            b = (numpy.sin(phi + 0.0*2.0*numpy.pi/3.0) + 1.0)/2.0

            result.append([r, g, b])
     
        return numpy.array(result)
    '''