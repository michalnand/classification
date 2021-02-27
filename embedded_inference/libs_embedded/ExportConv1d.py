from .Quantizer import *

def ExportConv1d(layer, layer_num, network_prefix, input_shape, quantization_type):
    
    layer_id = network_prefix + "_" + "layer_" + str(layer_num)

    weights = layer.weight.to("cpu").detach().numpy()
    bias    = layer.bias.to("cpu").detach().numpy()

    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        weights_quant, bias_quant, scale = Quantizer(weights, bias, max_value)

        weights_quant   = numpy.round(weights_quant, 0).astype(int)
        bias_quant      = numpy.round(bias_quant, 0).astype(int)

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int16_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        weights_quant, bias_quant, scale = Quantizer(weights, bias, max_value)

        weights_quant   = numpy.round(weights_quant, 0).astype(int)
        bias_quant      = numpy.round(bias_quant, 0).astype(int)

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

        scale           = 1024

        weights_quant   = weights
        bias_quant      = bias


    kernel_shape    = weights.shape
    kernel_padding  = layer.padding[0]
    kernel_stride   = layer.stride[0]

    var_weights = layer_id + "_weights"
    var_bias    = layer_id + "_bias"

    kernel_shape = weights.shape

    output_channels = kernel_shape[0]
    input_width     = input_shape[1]
    
    input_channels  = kernel_shape[1]
    kernel_size     = kernel_shape[2]

    output_width    = (input_width + 2*kernel_padding - (kernel_size - 1) - 1)//kernel_stride + 1

    output_shape    = (output_channels, output_width)

    #layer call code
    
    var_weights = layer_id + "_weights"
    var_bias    = layer_id + "_bias"

    code_network = "\tConv1d"

    code_network+= "<" + str(input_width) + ", " + str(input_channels) + ", " + str(output_channels) + ", "
    code_network+= str(kernel_size) + ", " + str(kernel_stride) + ", " + str(kernel_padding) + ", "
    code_network+= io_data_type + ", " + w_data_type + ", " + acc_data_type + ", "
    code_network+= str(max_value)  + ", "  + str(scale)
    code_network+= ">"

    code_network+= "(\n\t\toutput_buffer(), input_buffer(), \n"
    code_network+= "\t\t" + var_weights + ", " + var_bias + ");\n"

    code_network+= "\tswap_buffer();" + "\n\n"

    print("\n\n\n", weights.shape, "\n\n\n\n")
    #weights
    code_weight = "const " + w_data_type + " " + layer_id + "_weights[] = {" + "\n"
    for k in range(output_channels):
        for kw in range(kernel_size):
            for ch in range(input_channels):
                code_weight+= str(weights_quant[k][ch][kw]) + ", " 
                    
            if ch > 1:
                code_weight+= "\n"

        if kernel_shape[1] == 1:
            code_weight+= "\n"

    code_weight+= "};\n\n"
    
    #bias
    code_weight+= "const " + w_data_type + " " + layer_id + "_bias[] = {" + "\n"
    for i in range(output_channels):
        code_weight+= str(bias_quant[i]) + ", " 
    code_weight+= "};\n\n\n"


    code = (code_network, code_weight)
    
    required_memory       = max(input_shape[0]*input_shape[1], output_shape[0]*output_shape[1])

    macs = output_channels*kernel_size*input_channels*output_width #convolution
    macs+= output_channels*output_width    #bias


    print("export_Conv1d :")
    print("quantization    ", quantization_type)
    print("output_channels ", output_channels)
    print("input_width     ", input_width)
    print("input_channels  ", input_channels)
    print("kernel_size     ", kernel_size)
    print("stride          ", kernel_stride)
    print("padding         ", kernel_padding)
    print("output_shape    ", output_shape)
    print("macs            ", macs)
    print("\n\n")

    return code, output_shape, required_memory, macs