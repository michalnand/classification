import torch
import numpy


from .ExportLinear          import *
from .ExportConv1d          import *
from .ExportConv2d          import *
from .ExportGRU             import *
from .ExportReLU            import *
from .ExportGlobalAvgPool   import *

class ExportModel:

    def __init__(self, network_prefix, input_shape, output_shape, Model, pretrained_path, export_path, quantization_type, reccurent_stream_model = False):

        self.network_prefix = network_prefix
        self.input_shape    = input_shape
        self.output_shape   = output_shape

        self.reccurent_stream_model = reccurent_stream_model

        model = Model.Create(input_shape, output_shape)

        if pretrained_path is not None:
            model.load(pretrained_path)
            
        self.export_model(model, quantization_type)
        self.export_files(export_path, quantization_type)

   
    def export_model(self, model, quantization_type):
        layer_input_shape   = self.input_shape

        total_required_memory = numpy.prod(layer_input_shape)
        total_macs            = 0

        code_network = ""
        code_weights = ""

        if hasattr(model, "layers"):
            layers = model.layers
        else:
            layers = model.children()


        for i, layer in enumerate(layers):
            
            if isinstance(layer, torch.nn.Linear):                
                code, output_shape, required_memory, macs = ExportLinear(layer, i, self.network_prefix, layer_input_shape, quantization_type)

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.Conv1d):                
                code, output_shape, required_memory, macs = ExportConv1d(layer, i, self.network_prefix, layer_input_shape, quantization_type)

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.Conv2d):                
                code, output_shape, required_memory, macs = ExportConv2d(layer, i, self.network_prefix, layer_input_shape, quantization_type)

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.GRU):   
                if self.reccurent_stream_model:             
                    code, output_shape, required_memory, macs, rnn_hidden_size = ExportGRUStream(layer, i, self.network_prefix, layer_input_shape, quantization_type)
                    self.rnn_hidden_size = rnn_hidden_size
                else:
                    code, output_shape, required_memory, macs = ExportGRU(layer, i, self.network_prefix, layer_input_shape, quantization_type)

                code_network+= code[0]
                code_weights+= code[1]

            elif isinstance(layer, torch.nn.ReLU):                
                code, output_shape, required_memory, macs = ExportReLU(layer, i, self.network_prefix, layer_input_shape, quantization_type)

                code_network+= code[0]
                total_macs+= macs

            else:
                required_memory = 0
                macs            = 0
                output_shape    = layer_input_shape
                print("skipping unknow layer : ", layer, "\n\n")

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
        if self.reccurent_stream_model:
            self.code_h+= "\t\t " + "void reset();\n"
        self.code_h+= "\t\t " + "void forward();\n"

        if self.reccurent_stream_model:
            self.code_h+= "\tprivate:\n"
            self.code_h+= "\t\t " + "float *hidden_state;\n"


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

        if self.reccurent_stream_model:
            self.code_cpp+= "\t"    + "hidden_state = new float[" + str(self.rnn_hidden_size) + "];\n"
            self.code_cpp+= "\t"    + "reset();\n"
            
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


        self.code_cpp+= "void " + self.network_prefix + "::" + "reset()\n"
        self.code_cpp+= "{\n"
        self.code_cpp+= "\t"    + "for (unsigned int i = 0; i < " + str(self.rnn_hidden_size) + "; i++)\n"
        self.code_cpp+= "\t\t"  + "hidden_state[i] = 0;\n"
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
