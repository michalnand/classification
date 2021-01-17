import torch

class ModelQuantized(torch.nn.Module):
    def __init__(self, input_shape, output_shape, Model, pretrained_path):
        super(ModelQuantized, self).__init__()

        self.quant  = torch.quantization.QuantStub()

        self.model  = Model.Create(input_shape, output_shape)
        self.model.load(pretrained_path)
        self.model.eval()
        
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


    def get_fuse_list(self):
        result = []
 
        for i in range(len(self.model.layers)):
            tmp = []
            if isinstance(self.model.layers[i], torch.nn.Conv1d):
                tmp.append("model.model." + str(i))
                tmp.append("model.model." + str(i+1))
            elif isinstance(self.model.layers[i], torch.nn.Conv2d):
                tmp.append("model.model." + str(i))
                tmp.append("model.model." + str(i+1))
           
            if len(tmp) != 0:
                result.append(tmp)

        return result

    '''
    def fuse(self):
        print(self.model.model, "\n\n\n")
        #torch.quantization.fuse_modules(self.model.model, [['0']]) 
        torch.quantization.fuse_modules(self.model.model, [['1']]) 
    '''

def ModelQuantizer(input_shape, output_shape, Model, pretrained_path, dataset_sample_fp32, dataset_sample_y):
    model_fp32 = ModelQuantized(input_shape, output_shape, Model, pretrained_path)
    model_fp32.eval()

    '''
    print("\n\n\n\n")
    print(model_fp32.model.model[0].weight.data)
    print(model_fp32.model.model[0].bias.data)
    print("\n\n\n\n")  
    '''

    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    #fuse_list = model_fp32.get_fuse_list()
    #model_fp32_fused = torch.quantization.fuse_modules(model_fp32, fuse_list, inplace=False) 
    
    model_fp32_fused = model_fp32
    
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

    y_fp32 = model_fp32_prepared(dataset_sample_fp32)

    model_int8 = torch.quantization.convert(model_fp32_prepared)

    '''
    print("\n\n\n\n")
    print(model_int8.model)
    print(model_int8.model.model[0].scale)
    print(model_int8.model.model[0].zero_point)
    print(model_int8.model.model[0].weight().int_repr().data)
    print(model_int8.model.model[0].bias().int_repr().data)
    print(model_int8.model.model[0].bias())
    print("\n\n\n\n")  
    '''
    
    # run the model, relevant calculations will happen in int8
    y_int8 = model_int8(dataset_sample_fp32)

   
    indices_truth   = torch.argmax(dataset_sample_y, dim=1)
    indices_fp32    = torch.argmax(y_fp32, dim=1)
    indices_int8    = torch.argmax(y_int8, dim=1)


    total_count     = len(indices_truth)
    hit_fp32        = torch.sum(indices_truth == indices_fp32).detach().numpy()
    hit_int8        = torch.sum(indices_truth == indices_int8).detach().numpy()

    fp32_accuracy   = round((hit_fp32/total_count)*100.0, 3)
    int8_accuracy   = round((hit_int8/total_count)*100.0, 3)

    print("\n\n\n\n")
    print("quantization result : \n")
    print("fp32 accuracy = ", fp32_accuracy, "[%]")
    print("int8 accuracy = ", int8_accuracy, "[%]")
    print("\n\n\n\n")

    return model_fp32, model_int8
