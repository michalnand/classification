import torch
import numpy


def export_Linear(network_prefix, layer_num, quantization_type, input_shape, weights, bias, scale):        
    layer_id = network_prefix + "_" + "layer_" + str(layer_num)

    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 32768-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

        weights_quant   = weights
        bias_quant      = bias


    output_size     = weights.shape[0]
    input_size      = weights.shape[1]

    var_weights = layer_id + "_weights"
    var_bias    = layer_id + "_bias"

    code_network = ""

    #flatten code
    if len(input_shape) == 2:
        code_network+= "\tChannelReorder<" + io_data_type + ", " + str(input_shape[0]) + ", " + "1, " + str(input_shape[1]) + ">(output_buffer(), input_buffer());\n"
        code_network+= "\tswap_buffer();" + "\n\n"

    if len(input_shape) == 3:
        code_network+= "\tChannelReorder<" + io_data_type + ", " + str(input_shape[0]) + ", " + str(input_shape[1]) + ", " + str(input_shape[1]) + ">(output_buffer(), input_buffer());\n"
        code_network+= "\tswap_buffer();" + "\n\n"
    
    #layer call code

    code_network+= "\tLinear<" + str(input_size) + ", " + str(output_size) + ", " 
    code_network+= io_data_type + ", " + w_data_type + ", " + acc_data_type + ", "
    code_network+= str(max_value)  + ", " + str(scale)
    code_network+= ">"

    code_network+= "(\n\t\toutput_buffer(), input_buffer(), \n"
    code_network+= "\t\t" + var_weights + ", " + var_bias + ");\n"

    code_network+= "\tswap_buffer();" + "\n\n"

    #weights
    code_weight = "const " + w_data_type + " " + layer_id + "_weights[] = {" + "\n"
    for j in range(output_size):
        for i in range(input_size):
            code_weight+= str(weights_quant[j][i]) + ", " 
                
        code_weight+= "\n"

    code_weight+= "};\n\n"
    
    #bias
    code_weight+= "const " + w_data_type + " " + layer_id + "_bias[] = {" + "\n"
    for i in range(len(bias)):
        code_weight+= str(bias_quant[i]) + ", " 
    code_weight+= "};\n\n\n"


    code = (code_network, code_weight)
    macs = output_size*input_size + output_size

    print("export_Linear :")
    print("output_size    ", output_size)
    print("input_size     ", input_size)
    print("macs           ", macs)
    print("\n\n")


    return code, (output_size, ), output_size, macs

def export_Conv1d(network_prefix, layer_num, quantization_type, input_shape, weights, bias, scale, kernel_stride, kernel_padding):        
    layer_id = network_prefix + "_" + "layer_" + str(layer_num)

    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 32768-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

        weights_quant   = weights
        bias_quant      = bias

    kernel_shape    = weights.shape

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


def export_Conv2d(network_prefix, layer_num, quantization_type, input_shape, weights, bias, scale, kernel_stride, kernel_padding):        
    layer_id = network_prefix + "_" + "layer_" + str(layer_num)

    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 32768-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

        weights_quant   = weights
        bias_quant      = bias


    kernel_shape    = weights.shape

    var_weights = layer_id + "_weights"
    var_bias    = layer_id + "_bias"


    output_channels = kernel_shape[0]
    input_height    = input_shape[1]
    input_width     = input_shape[2]
    input_channels  = kernel_shape[1]
    kernel_size     = kernel_shape[2]

    height_         = (input_height + 2*kernel_padding - (kernel_size - 1) - 1)//kernel_stride + 1
    width_          = (input_width  + 2*kernel_padding - (kernel_size - 1) - 1)//kernel_stride + 1

    output_shape    = (output_channels, height_, width_)

    #layer call code
    
    var_weights = layer_id + "_weights"
    var_bias    = layer_id + "_bias" 

    code_network = "\tConv2d"

    code_network+= "<" + str(input_height) + ", " + str(input_width) + ", " + str(input_channels) + ", " + str(output_channels) + ", "
    code_network+= str(kernel_size) + ", " + str(kernel_stride) + ", "  + str(kernel_padding) + ", "
    code_network+= io_data_type + ", " + w_data_type + ", " + acc_data_type + ", "
    code_network+= str(max_value)  + ", " + str(scale)
    code_network+= ">"

    code_network+= "(\n\t\toutput_buffer(), input_buffer(), \n"
    code_network+= "\t\t" + var_weights + ", " + var_bias + ");\n"
    

    code_network+= "\tswap_buffer();" + "\n\n"

    #weights
    code_weight = "const " + w_data_type + " " + layer_id + "_weights[] = {" + "\n"
    for k in range(kernel_shape[0]):
        for kh in range(kernel_shape[2]):
            for kw in range(kernel_shape[3]):
                for ch in range(kernel_shape[1]):
                    code_weight+= str(weights_quant[k][ch][kh][kw]) + ", " 
                    
                if ch > 1:
                    code_weight+= "\n"

        if kernel_shape[1] == 1:
            code_weight+= "\n"

    code_weight+= "};\n\n"
    
    #bias
    code_weight+= "const " + w_data_type + " " + layer_id + "_bias[] = {" + "\n"
    for i in range(len(bias)):
        code_weight+= str(bias_quant[i]) + ", " 
    code_weight+= "};\n\n\n"


    code = (code_network, code_weight)
    
    required_memory       = output_shape[0]*output_shape[1]*output_shape[2]

    macs = output_channels*(kernel_size**2)*input_channels*output_shape[1]*output_shape[2] #convolution
    macs+= output_channels*output_shape[1]*output_shape[2]    #bias


    print("export_Conv2d :")
    print("output_channels ", output_channels)
    print("input_height    ", input_height)
    print("input_width     ", input_width)
    print("input_channels  ", input_channels)
    print("kernel_size     ", kernel_size)
    print("stride          ", kernel_stride)
    print("padding         ", kernel_padding)
    print("output_shape    ", output_shape)
    print("macs            ", macs)
    print("\n\n")

    return code, output_shape, required_memory, macs


def export_ReLU(network_prefix, layer_num, quantization_type, input_shape):
    
    output_shape = input_shape

    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 32768-1

       
    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

       

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


def export_AvgPool1d(layer, input_shape, layer_num, quantization_type):
    
    channels    = input_shape[0]
    width       = input_shape[1]
    
    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 32768-1

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0


    output_shape = (1, 1, channels)

    code_network = "\tGlobalAveragePooling1d<"
    code_network+= str(1) + ", " + str(width) + ", " + str(channels) + ", " + str(io_data_type) + ", " + str(acc_data_type) + ", " + str(max_value) + ">(\n"
    code_network+= "\t\toutput_buffer(), input_buffer());\n" 
    code_network+= "\tswap_buffer();" + "\n\n"

    code = (code_network, "", "")

    macs = width*channels*2

    size = width*channels

    print("export_AvgPool1d :")
    print("IO shape        ", output_shape)
    print("macs            ", macs)
    print("\n\n")
    
    return code, output_shape, size, macs





def export_LSTM(layer, input_shape, layer_num, quantization_type):
    
    channels    = input_shape[0]
    width       = input_shape[1]

    w_ii, w_if, w_ic, w_io = layer.weight_ih_l0.chunk(4, 0)
    w_hi, w_hf, w_hc, w_ho = layer.weight_hh_l0.chunk(4, 0)

    w_ii, w_if, w_ic, w_io = layer.bias_ih_l0.chunk(4, 0)
    w_hi, w_hf, w_hc, w_ho = layer.bias_hh_l0.chunk(4, 0)
    
    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 32768-1

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0


    output_shape = (1, 1, channels)

    code_network = "\tGlobalAveragePooling1d<"
    code_network+= str(1) + ", " + str(width) + ", " + str(channels) + ", " + str(io_data_type) + ", " + str(acc_data_type) + ", " + str(max_value) + ">(\n"
    code_network+= "\t\toutput_buffer(), input_buffer());\n" 
    code_network+= "\tswap_buffer();" + "\n\n"

    code = (code_network, "", "")

    macs = width*channels*2

    size = width*channels

    print("export_AvgPool1d :")
    print("IO shape        ", output_shape)
    print("macs            ", macs)
    print("\n\n")
    
    return code, output_shape, size, macs








def export_LSTM(network_prefix, layer_num, quantization_type, input_shape, weights, bias, scale):        
    layer_id = network_prefix + "_" + "layer_" + str(layer_num)

    #extract LSTM layer weights
    w_ii, w_if, w_ic, w_io = layer.weight_ih_l0.chunk(4, 0)
    w_hi, w_hf, w_hc, w_ho = layer.weight_hh_l0.chunk(4, 0)

    b_ii, b_if, b_ic, b_io = layer.bias_ih_l0.chunk(4, 0)
    b_hi, b_hf, b_hc, b_ho = layer.bias_hh_l0.chunk(4, 0)
    
    #compute output
    #input  : x, hidden state h, cell state c
    #output : new hidden state h, new cell state c

    input       = sigmoid(w_ii*x + b_ii + w_hi*h + b_hi)
    forget      = sigmoid(w_if*x + b_if + w_hf*h + b_hf)
    candidate   = tanh(w_ic*x + b_ic + w_hc*h + b_hc)
    output      = sigmoid(w_io*x + b_io + w_ho*h + b_ho)

    c           = forget*c + input*candidate
    h           = output*tanh(c)

    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    elif quantization_type == "int16":
        io_data_type    = "int16_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 32768-1

        weights_quant   = numpy.round(weights, 0).astype(int)
        bias_quant      = numpy.round(bias, 0).astype(int)

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

        weights_quant   = weights
        bias_quant      = bias


    output_size     = weights.shape[0]
    input_size      = weights.shape[1]

    var_weights = layer_id + "_weights"
    var_bias    = layer_id + "_bias"

    code_network = ""

    #flatten code
    if len(input_shape) == 2:
        code_network+= "\tChannelReorder<" + io_data_type + ", " + str(input_shape[0]) + ", " + "1, " + str(input_shape[1]) + ">(output_buffer(), input_buffer());\n"
        code_network+= "\tswap_buffer();" + "\n\n"

    if len(input_shape) == 3:
        code_network+= "\tChannelReorder<" + io_data_type + ", " + str(input_shape[0]) + ", " + str(input_shape[1]) + ", " + str(input_shape[1]) + ">(output_buffer(), input_buffer());\n"
        code_network+= "\tswap_buffer();" + "\n\n"
    
    #layer call code

    code_network+= "\tLinear<" + str(input_size) + ", " + str(output_size) + ", " 
    code_network+= io_data_type + ", " + w_data_type + ", " + acc_data_type + ", "
    code_network+= str(max_value)  + ", " + str(scale)
    code_network+= ">"

    code_network+= "(\n\t\toutput_buffer(), input_buffer(), \n"
    code_network+= "\t\t" + var_weights + ", " + var_bias + ");\n"

    code_network+= "\tswap_buffer();" + "\n\n"

    #weights
    code_weight = "const " + w_data_type + " " + layer_id + "_weights[] = {" + "\n"
    for j in range(output_size):
        for i in range(input_size):
            code_weight+= str(weights_quant[j][i]) + ", " 
                
        code_weight+= "\n"

    code_weight+= "};\n\n"
    
    #bias
    code_weight+= "const " + w_data_type + " " + layer_id + "_bias[] = {" + "\n"
    for i in range(len(bias)):
        code_weight+= str(bias_quant[i]) + ", " 
    code_weight+= "};\n\n\n"


    code = (code_network, code_weight)
    macs = output_size*input_size + output_size

    print("export_Linear :")
    print("output_size    ", output_size)
    print("input_size     ", input_size)
    print("macs           ", macs)
    print("\n\n")


    return code, (output_size, ), output_size, macs
