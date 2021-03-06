import numpy
import torch
import colorsys

class SegmentationInference:
    def __init__(self, Model, model_pretrained_path):
        channels            = 3
        height              = 512
        width               = 1024

        self.model      = Model.Create((channels, height, width), (self.classes_count, height, width))

        if model_pretrained_path is not None:
            self.model.load(model_pretrained_path)
        self.model.eval()

        image_t         = torch.zeros((channels, height, width)).unsqueeze(0).to(self.model.device).float()
        prediction_t    = self.model(image_t)

        self.classes_count = prediction_t.shape[1]

        self.colors     = self._make_colors(self.classes_count)


    def process(self, image_np, channel_first = False, alpha = 0.5):
        
        image_np        = image_np/256.0

        prediction_np   = self._predict(image_np, channel_first)

        mask            = self.colors[prediction_np, :]
        result          = (1.0 - alpha)*image_np + alpha*mask

        return prediction_np, mask, result

    def _predict(self, image_np, channel_first):

        if channel_first == False:
            image_np    = numpy.moveaxis(image_np, 2, 0)

        image_t     = torch.from_numpy(image_np).unsqueeze(0).to(self.model.device).float()

        prediction_t    = self.model(image_t).squeeze(0)
        prediction_t    = torch.argmax(prediction_t, dim=0)
        prediction_np   = prediction_t.detach().to("cpu").numpy()

        return prediction_np

    def _make_colors(self, count):

        hsv_tuples = []
        for i in range(count):
            j   = i//4
            
            h = j*1.0/(count//4)
            s = (i%4)/4.0 + 0.25
            v = 0.5 
            
            hsv_tuples.append([h, s, v])

        result = []
        for i in range(count):
            result.append(colorsys.hsv_to_rgb(hsv_tuples[i][0], hsv_tuples[i][1], hsv_tuples[i][2]))

     
        return numpy.array(result)


