from .Quantizer import *

def ExportReLU(layer, layer_num, network_prefix, input_shape, quantization_type):
    if quantization_type == "int8":
        io_data_type    = "int8_t"
    elif quantization_type == "int16":
        io_data_type    = "int16_t"
    else:
        io_data_type    = "float"

    output_shape    = input_shape

    size            = numpy.prod(output_shape)
    macs            = 4*size
    
    code_network    = "\tReLU<" + io_data_type + ", " + str(size) + ">(" + "output_buffer(), input_buffer());\n"
    code_network+=  "\tswap_buffer();" + "\n\n"

    code = (code_network, "", "")

    print("export_ReLU :")
    print("IO shape        ", output_shape)
    print("macs            ", macs)
    print("\n\n")
    
    return code, output_shape, size, macs

