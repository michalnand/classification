import torch
import numpy

from .post_training_quantization import *
from .export_layer import *


class ExportModel:

    def __init__(self, network_prefix, input_shape, output_shape, Model, pretrained_path, export_path, quantization_type):

        self.network_prefix = network_prefix
        self.input_shape    = input_shape
        self.output_shape   = output_shape

        model = Model.Create(input_shape, output_shape)
        model.load(pretrained_path)

        if quantization_type == "float":
            self.export_model_fp32(model)
        elif quantization_type == "int8":
            self.export_model_int(model, "int8")
        elif quantization_type == "int16":
            self.export_model_int(model, "int16")
        
        self.export_files(export_path, quantization_type)

   
    def export_model_int(self, model, quantization_type):


        layer_input_shape = self.input_shape

        total_required_memory = numpy.prod(layer_input_shape)
        total_macs            = 0

        code_network = ""
        code_weights = ""


        r = 127

        for i in range(len(model.layers)):
            layer = model.layers[i]
            
            if isinstance(layer, torch.nn.Linear):
                weights_quant, bias_quant, scale = self._quantize(layer.weight.to("cpu").detach().numpy(), layer.bias.to("cpu").detach().numpy(), r)
                
                code, output_shape, required_memory, macs = export_Linear(self.network_prefix, i, quantization_type, layer_input_shape, weights_quant, bias_quant, scale)

                code_network+= code[0]
                code_weights+= code[1]


            elif isinstance(layer, torch.nn.Conv1d):
                weights_quant, bias_quant, scale = self._quantize(layer.weight.to("cpu").detach().numpy(), layer.bias.to("cpu").detach().numpy(), r)
             
                code, output_shape, required_memory, macs = export_Conv1d(self.network_prefix, i, quantization_type, layer_input_shape, weights_quant, bias_quant, scale, layer.stride[0], layer.padding[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.Conv2d):
                weights_quant, bias_quant, scale = self._quantize(layer.weight.to("cpu").detach().numpy(), layer.bias.to("cpu").detach().numpy(), r)
                
                code, output_shape, required_memory, macs = export_Conv2d(self.network_prefix, i, quantization_type, layer_input_shape, weights_quant, bias_quant, scale, layer.stride[0], layer.padding[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.ReLU):
                code, output_shape, required_memory, macs = export_ReLU(self.network_prefix, i, quantization_type, layer_input_shape)

                code_network+= code[0]

                total_macs+= macs

            elif isinstance(layer, torch.nn.AvgPool1d):
                code, output_shape, required_memory, macs = export_AvgPool1d(layer, layer_input_shape, i, quantization_type)

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
                weight_np   = 1.0*layer.weight.detach().to("cpu").numpy()
                bias_np     = 1.0*layer.bias.detach().to("cpu").numpy()
                
                code, output_shape, required_memory, macs = export_Linear(self.network_prefix, i, "float", layer_input_shape, weight_np, bias_np, scale_quant)

                code_network+= code[0]
                code_weights+= code[1]


            elif isinstance(layer, torch.nn.Conv1d):
                weight_np   = 1.0*layer.weight.detach().to("cpu").numpy()
                bias_np     = 1.0*layer.bias.detach().to("cpu").numpy()

                code, output_shape, required_memory, macs = export_Conv1d(self.network_prefix, i, "float", layer_input_shape, weight_np, bias_np, scale_quant, layer.stride[0], layer.padding[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.Conv2d):
                weight_np   = 1.0*layer.weight.detach().to("cpu").numpy()
                bias_np     = 1.0*layer.bias.detach().to("cpu").numpy()

                code, output_shape, required_memory, macs = export_Conv2d(self.network_prefix, i, "float", layer_input_shape, weight_np, bias_np, scale_quant, layer.stride[0], layer.padding[0])

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.ReLU):
                code, output_shape, required_memory, macs = export_ReLU(self.network_prefix, i, "float", layer_input_shape)

                code_network+= code[0]

                total_macs+= macs

            elif isinstance(layer, torch.nn.AvgPool1d):
                code, output_shape, required_memory, macs = export_AvgPool1d(layer, layer_input_shape, i, "float")

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



    def export_files(self, export_path, quantization_type):

        if quantization_type == "int8":
            io_type = "int8_t"
        elif quantization_type == "int16":
            io_type = "int16_t"
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

    
    def _quantize(self, weights, bias, r = 127):

        tmp             = numpy.concatenate([weights.flatten(), bias.flatten()])

        #scale           = 4.0*numpy.std(tmp) 
        scale           = numpy.max(numpy.abs(tmp))

 
        print("layer stats")
        print("mean = ", tmp.mean())
        print("std =  ", tmp.std())
        print("max =  ", tmp.max())
        print("min =  ", tmp.min())
        print("scale =", scale)
        print("\n")

        weights_scaled  = weights/scale
        bias_scaled     = bias/scale
              
        weights_quant   = numpy.clip(r*weights_scaled,    -r, r)
        bias_quant      = numpy.clip(r*bias_scaled,       -r, r)
        

        return weights_quant, bias_quant, int(scale*1024)