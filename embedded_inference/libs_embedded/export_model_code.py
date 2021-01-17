import torch
import numpy

from .post_training_quantization import *
from .export_layer import *



class ExportModelCode:

    def __init__(self, network_prefix, input_shape, output_shape, Model, pretrained_path, export_path, dataset_sample_fp32, dataset_sample_y, int8_export = True):

        self.network_prefix = network_prefix
        self.input_shape    = input_shape
        self.output_shape   = output_shape

        model_fp32 = Model.Create(input_shape, output_shape)
        model_fp32.load(pretrained_path)
        
        model_fp32, model_int8 = ModelQuantizer(input_shape, output_shape, Model, pretrained_path, dataset_sample_fp32, dataset_sample_y)

        if int8_export:
            self.export_model_int8(model_int8.model)
        else:            
            self.export_model_fp32(model_fp32.model)

        #self.export_model_int8_custom(model_fp32)

        self.export_files(export_path, int8_export)

    def export_model_int8(self, model):

        scale_quant = 1024

        layer_input_shape = self.input_shape

        total_required_memory = numpy.prod(layer_input_shape)
        total_macs            = 0

        code_network = ""
        code_weights = ""

        for i in range(len(model.model)):
            layer = model.model[i]
            
            if isinstance(layer, torch.nn.quantized.Linear):
                
                scale           = layer.scale
                weights_quant   = layer.weight().int_repr().data.detach().to("cpu").numpy()
                bias_quant      = (layer.bias()/scale).detach().to("cpu").numpy()

                scale_quant     = int(scale*scale_quant)
                zero_quant      = int(layer.zero_point/(scale*127))


                code, output_shape, required_memory, macs =  export_Linear(self.network_prefix, i, True, layer_input_shape, weights_quant, bias_quant, scale_quant, zero_quant)
                
                code_network+= code[0]
                code_weights+= code[1]


            elif isinstance(layer, torch.nn.quantized.Conv1d):

                scale           = layer.scale
                weights_quant   = layer.weight().int_repr().data.detach().to("cpu").numpy()
                bias_quant      = (layer.bias()/scale).detach().to("cpu").numpy()

                scale_quant     = int(scale*scale_quant)
                zero_quant      = int(layer.zero_point/(scale*127))


                code, output_shape, required_memory, macs = export_Conv1d(self.network_prefix, i, True, layer_input_shape, weights_quant, bias_quant, scale_quant, zero_quant, layer.stride[0])
               
                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.quantized.Conv2d):
                scale           = layer.scale
                weights_quant   = layer.weight().int_repr().data.detach().to("cpu").numpy()
                bias_quant      = (layer.bias()/scale).detach().to("cpu").numpy()

                scale_quant     = int(scale*scale_quant)
                zero_quant      = int(layer.zero_point/(scale*127))

                code, output_shape, required_memory, macs = export_Conv2d(self.network_prefix, i, True, layer_input_shape, weights_quant, bias_quant, scale_quant, zero_quant, layer.stride[0])
               
                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.quantized.ReLU):
                code, output_shape, required_memory, macs = export_ReLU(self.network_prefix, i, True, layer_input_shape)

                code_network+= code[0]

                total_macs+= macs

            elif isinstance(layer, torch.nn.AvgPool1d):
                code, output_shape, required_memory, macs = export_AvgPool1d(layer, layer_input_shape, i, True)

                code_network+= code[0]

                total_macs+= macs

            else:
                required_memory = 0
                macs            = 0
                output_shape    = layer_input_shape

            
            layer_input_shape = output_shape

            
            total_required_memory = max(total_required_memory, required_memory)
            total_macs+= macs


        self.code_network   = code_network
        self.code_weights   = code_weights
        self.total_required_memory = total_required_memory 
        self.total_macs     = total_macs

    def export_model_int8_custom(self, model):


        layer_input_shape = self.input_shape

        total_required_memory = numpy.prod(layer_input_shape)
        total_macs            = 0

        code_network = ""
        code_weights = ""

        for i in range(len(model.layers)):
            layer = model.layers[i]
            
            if isinstance(layer, torch.nn.Linear):
                weights_quant, bias_quant, scale, zero_point = self._quantize(layer.weight.to("cpu").detach().numpy(), layer.bias.to("cpu").detach().numpy())
                
                code, output_shape, required_memory, macs = export_Linear(self.network_prefix, i, True, layer_input_shape, weights_quant, bias_quant, scale, zero_point)

                code_network+= code[0]
                code_weights+= code[1]


            elif isinstance(layer, torch.nn.Conv1d):
                weights_quant, bias_quant, scale, zero_point = self._quantize(layer.weight.to("cpu").detach().numpy(), layer.bias.to("cpu").detach().numpy())
             
                code, output_shape, required_memory, macs = export_Conv1d(self.network_prefix, i, True, layer_input_shape, weights_quant, bias_quant, scale, zero_point, layer.stride[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.Conv2d):
                weights_quant, bias_quant, scale, zero_point = self._quantize(layer.weight.to("cpu").detach().numpy(), layer.bias.to("cpu").detach().numpy())
                
                code, output_shape, required_memory, macs = export_Conv2d(self.network_prefix, i, True, layer_input_shape, weights_quant, bias_quant, scale, zero_point, layer.stride[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.ReLU):
                code, output_shape, required_memory, macs = export_ReLU(self.network_prefix, i, True, layer_input_shape)

                code_network+= code[0]

                total_macs+= macs

            elif isinstance(layer, torch.nn.AvgPool1d):
                code, output_shape, required_memory, macs = export_AvgPool1d(layer, layer_input_shape, i, True)

                code_network+= code[0]

                total_macs+= macs

            else:
                required_memory = 0
                macs            = 0
                output_shape    = layer_input_shape

            
            layer_input_shape = output_shape

            
            total_required_memory = max(total_required_memory, required_memory)
            total_macs+= macs


        self.code_network   = code_network
        self.code_weights   = code_weights
        self.total_required_memory = total_required_memory 
        self.total_macs     = total_macs

    def export_model_fp32(self, model):

        scale_quant = 1024

        layer_input_shape = self.input_shape

        total_required_memory = numpy.prod(layer_input_shape)
        total_macs            = 0

        code_network = ""
        code_weights = ""

        for i in range(len(model.layers)):
            layer = model.layers[i]
            
            if isinstance(layer, torch.nn.Linear):
                code, output_shape, required_memory, macs = export_Linear(self.network_prefix, i, False, layer_input_shape, layer.weight, layer.bias, scale_quant, 0)

                code_network+= code[0]
                code_weights+= code[1]


            elif isinstance(layer, torch.nn.Conv1d):
                code, output_shape, required_memory, macs = export_Conv1d(self.network_prefix, i, False, layer_input_shape, layer.weight, layer.bias, scale_quant, 0, layer.stride[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.Conv2d):
                code, output_shape, required_memory, macs = export_Conv2d(self.network_prefix, i, False, layer_input_shape, layer.weight, layer.bias, scale_quant, 0, layer.stride[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.ReLU):
                code, output_shape, required_memory, macs = export_ReLU(self.network_prefix, i, False, layer_input_shape)

                code_network+= code[0]

                total_macs+= macs

            elif isinstance(layer, torch.nn.AvgPool1d):
                code, output_shape, required_memory, macs = export_AvgPool1d(layer, layer_input_shape, i, False)

                code_network+= code[0]

                total_macs+= macs

            else:
                required_memory = 0
                macs            = 0
                output_shape    = layer_input_shape

            
            layer_input_shape = output_shape

            
            total_required_memory = max(total_required_memory, required_memory)
            total_macs+= macs


        self.code_network   = code_network
        self.code_weights   = code_weights
        self.total_required_memory = total_required_memory 
        self.total_macs     = total_macs



    def export_files(self, export_path, int8_export):

        if int8_export:
            io_type = "int8_t"
        else:
            io_type = "float"

        self.code_h = ""
        self.code_h+= "#ifndef _" + self.network_prefix + "_H_\n"
        self.code_h+= "#define _" + self.network_prefix + "_H_\n"
        self.code_h+= "\n\n"
        self.code_h+= "#include <ModelInterface.h>"
        self.code_h+= "\n\n"
        self.code_h+= "class " + self.network_prefix + " : public ModelInterface<"+ io_type +">\n"
        self.code_h+= "{\n"
        self.code_h+= "\tpublic:\n"
        self.code_h+= "\t\t " + self.network_prefix + "();\n"
        self.code_h+= "\t\t " + "void forward();\n"
        self.code_h+= "};\n"

        self.code_h+= "\n\n"
        self.code_h+= "#endif\n"


        self.code_cpp = ""
        self.code_cpp+= "#include <" + self.network_prefix + ".h>\n"
        self.code_cpp+= "\n\n"
        self.code_cpp+= self.code_weights + "\n\n"

        self.code_cpp+= self.network_prefix + "::" + self.network_prefix + "()\n"
        self.code_cpp+= "\t: ModelInterface()\n"
        self.code_cpp+= "{\n"
        self.code_cpp+= "\t" + "init_buffer("  + str(self.total_required_memory) + ");\n"
        self.code_cpp+= "\t" + "total_macs = " + str(self.total_macs) + ";\n"

        if len(self.input_shape) == 3:
            self.code_cpp+= "\t" + "input_channels = "  + str(self.input_shape[0]) + ";\n"
            self.code_cpp+= "\t" + "input_height = "    + str(self.input_shape[1]) + ";\n"
            self.code_cpp+= "\t" + "input_width = "     + str(self.input_shape[2]) + ";\n"
        elif len(self.input_shape) == 2:
            self.code_cpp+= "\t" + "input_channels = "  + str(self.input_shape[0]) + ";\n"
            self.code_cpp+= "\t" + "input_height = "    + str(1) + ";\n"
            self.code_cpp+= "\t" + "input_width = "     + str(self.input_shape[1]) + ";\n" 
        elif len(self.input_shape) == 1:
            self.code_cpp+= "\t" + "input_channels = "  + str(self.input_shape[0]) + ";\n"
            self.code_cpp+= "\t" + "input_height = "    + str(1) + ";\n"
            self.code_cpp+= "\t" + "input_width = "     + str(1) + ";\n" 

        self.code_cpp+= "\t" + "output_channels = "     + str(self.output_shape[0]) + ";\n"

        if len(self.output_shape) > 1:
            self.code_cpp+= "\t" + "output_height = "   + str(self.output_shape[1]) + ";\n"
        else:
            self.code_cpp+= "\t" + "output_height = "   + str(1) + ";\n"

        if len(self.output_shape) > 2:
            self.code_cpp+= "\t" + "output_width = "    + str(self.output_shape[2]) + ";\n"
        else:
            self.code_cpp+= "\t" + "output_width = "    + str(1) + ";\n"

        self.code_cpp+= "}\n\n"


        self.code_cpp+= "void " + self.network_prefix + "::" + "forward()\n"
        self.code_cpp+= "{\n"
        self.code_cpp+= self.code_network
        self.code_cpp+= "\tswap_buffer();" + "\n"
        self.code_cpp+= "}\n" 

        cpp_file = open(export_path + "/" + self.network_prefix + ".cpp", "w")
        cpp_file.write(self.code_cpp)
        cpp_file.close() 
                
        h_file = open(export_path + "/" + self.network_prefix + ".h", "w")
        h_file.write(self.code_h)
        h_file.close()

    
    def _quantize(self, weights, bias):

        tmp   = numpy.concatenate([weights.flatten(), bias.flatten()])

        #scale      = 2.0*numpy.std(tmp)
        scale       = numpy.max(numpy.abs(tmp))
        zero_point  = (tmp/scale).mean()

        weights_scaled  = weights/scale - zero_point
        bias_scaled     = bias/scale    - zero_point

        print(">>>> ", scale, zero_point, weights.mean(), weights_scaled.mean())
        
        weights_quant   = numpy.clip(127*weights_scaled, -127, 127)
        bias_quant      = numpy.clip(127*bias_scaled, -127, 127)
        

        return weights_quant, bias_quant, int(scale*1024), int(zero_point*127)