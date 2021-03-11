import numpy
import torch
import colorsys

if torch.cuda.is_available():
    from torch2trt import torch2trt

class SegmentationInference:
    def __init__(self, Model, model_pretrained_path, classes_count, height = 480, width = 640):
        print("creating model")
        channels        = 3
        self.model      = Model.Create((channels, height, width), (classes_count, height, width))
        self.device     = self.model.device

        if model_pretrained_path is not None:
            print("loading wights")
            self.model.load(model_pretrained_path)

        self.model.eval()

        if self.device == "cuda":
            print("converting model into torchRT")
            x = torch.ones((1, channels, height, width)).to("cuda")
            self.model = torch2trt(self.model, [x])

        self.colors     = self._make_colors(classes_count)

        print("SegmentationInference ready")


    def process(self, image_np, channel_first = False, alpha = 0.35):
        image_t = torch.from_numpy(image_np).float().to(self.device)
        image_t = image_t/256.0

        if channel_first == False:
            image_in_t     = image_t.transpose(0, 2).transpose(1, 2) 
        else:
            image_in_t     = image_t

        image_in_t = image_in_t

        prediction_t    = self.model(image_in_t.unsqueeze(0)).squeeze(0)
        prediction_t    = torch.argmax(prediction_t, dim=0)

        prediction_t    = prediction_t.transpose(0, 1)

        mask_t          = self.colors[prediction_t, :].transpose(0, 1)        
        
        result_t        = (1.0 - alpha)*image_t + alpha*mask_t

        prediction      = prediction_t.detach().to("cpu").numpy()
        mask            = mask_t.detach().to("cpu").numpy()
        result          = result_t.detach().to("cpu").numpy()

        return prediction, mask, result
        


    def _make_colors(self, count):

        result = []
        
        result.append([0, 0, 1])
        result.append([1, 1, 0])
        result.append([0, 1, 0])
        result.append([0, 1, 1])
        result.append([1, 0, 0])

        result = torch.from_numpy(numpy.array(result)).to(self.device)
        
        return result

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