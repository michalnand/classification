from .Quantizer import *

def ExportAvgPool1d(layer, layer_num, network_prefix, input_shape, quantization_type):
    if quantization_type == "int8":
        io_data_type    = "int8_t"
    else:
        io_data_type    = "float"


    channels    = input_shape[0]
    width       = input_shape[1]

    output_shape = (1, 1, channels)

    code_network = "\tGlobalAveragePooling1d<"
    code_network+= str(1) + ", " + str(width) + ", " + str(channels) + ", " + str(io_data_type) + ", " + str(acc_data_type) + ", " + str(max_value) + ">(\n"
    code_network+= "\t\toutput_buffer(), input_buffer());\n" 
    code_network+= "\tswap_buffer();" + "\n\n"

    code = (code_network, "", "")

    macs = width*channels*2

    size = width*channels

    print("export_AvgPool1d :")
    print("quantization     ", quantization_type)
    print("IO shape        ", output_shape)
    print("macs            ", macs)
    print("\n\n")
    
    return code, output_shape, size, macs