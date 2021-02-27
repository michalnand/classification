from .Quantizer import *

def ExportGlobalAvgPool(layer, layer_num, network_prefix, input_shape, quantization_type):
    if quantization_type == "int8":
        io_data_type    = "int8_t"
    elif quantization_type == "int16":
        io_data_type    = "int16_t"
    else:
        io_data_type    = "float"

    if len(input_shape == 2):
        channels    = input_shape[0]
        height      = 1
        width       = input_shape[1]
    else:
        channels    = input_shape[0]
        height      = input_shape[1]
        width       = input_shape[2]

    output_shape = (1, 1, channels)

    code_network = "\tGlobalAveragePool<"
    code_network+= str(height) + ", " + str(width) + ", " + str(channels) + ", " + str(io_data_type) + ", " + str(acc_data_type) + ", " + str(max_value) + ">(\n"
    code_network+= "\t\toutput_buffer(), input_buffer());\n" 
    code_network+= "\tswap_buffer();" + "\n\n"

    code = (code_network, "", "")

    macs = width*channels*2

    size = width*channels

    print("export_GlobalAvgPool :")
    print("quantization    ", quantization_type)
    print("input shape     ", input_shape)
    print("output shape    ", output_shape)
    print("macs            ", macs)
    print("\n\n")
    
    return code, output_shape, size, macs