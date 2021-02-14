import numpy
import torch

import onnxruntime

class SegmentationInference:
    def __init__(self, Model, model_pretrained_path, classes_count, use_onnx_model = True):
        channels            = 3
        height              = 480
        width               = 640

        self.use_onnx_model = use_onnx_model 

        print("creating model")
        self.model      = Model.Create((channels, height, width), (classes_count, height, width))

        if model_pretrained_path is not None:
            print("loading wights")
            self.model.load(model_pretrained_path)
        self.model.eval()

        if self.use_onnx_model:
            print("converting model into ONNX")
            x = torch.ones((1, channels, height, width)).to(self.model.device)

            # Export the model
            torch.onnx.export(self.model,               # model being run
                            x,                         # model input (or a tuple for multiple inputs)
                            "model.onnx",   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=10,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # 0 for variable lenght axes
                                            'output' : {0 : 'batch_size'}})

            self.onnx_model = onnxruntime.InferenceSession("model.onnx")

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

        if self.use_onnx_model:
            image_np        = numpy.expand_dims(image_np, 0).astype(numpy.float32)
            ort_inputs      = {self.onnx_model.get_inputs()[0].name: image_np}
            prediction_l    = self.onnx_model.run(None, ort_inputs)
            prediction_np   = numpy.squeeze(prediction_l[0], 0)
        else:
            image_t         = torch.from_numpy(image_np).unsqueeze(0).float().to(self.model.device)
            prediction_np   = self.model(image_t).squeeze(0).detach().to("cpu").numpy()
        
        prediction_np   = numpy.argmax(prediction_np, axis=0)

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