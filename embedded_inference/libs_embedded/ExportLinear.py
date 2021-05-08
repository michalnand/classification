from .Quantizer import *

def ExportLinear(layer, layer_num, network_prefix, input_shape, quantization_type):
    
    layer_id = network_prefix + "_" + "layer_" + str(layer_num)

    padding_inputs  = 4
    padding_outputs = 8

    weights_raw = layer.weight.to("cpu").detach().numpy()
    bias_raw    = layer.bias.to("cpu").detach().numpy()

    #add padding
    in_features  = weights_raw.shape[1]
    out_features = weights_raw.shape[0]

    p_out       = (padding_outputs - (out_features%padding_outputs))%padding_outputs
    p_in        = (padding_inputs  - (in_features%padding_inputs))%padding_inputs

    weights = numpy.zeros((out_features + p_out, in_features + p_in))
    weights[0:out_features, 0:in_features] = weights_raw

    bias        = numpy.zeros((out_features + p_out))
    bias[0:out_features] = bias_raw
    
    
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

    output_size     = weights.shape[0]
    input_size      = weights.shape[1]

    
    var_weights = layer_id + "_weights"
    var_bias    = layer_id + "_bias"

    code_network = ""

    #flatten code if linear layer after convolutional
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
    print("quantization   ", quantization_type)
    print("output_size    ", output_size)
    print("input_size     ", input_size)
    print("macs           ", macs)
    print("\n\n")


    return code, (output_size, ), output_size*4, macs