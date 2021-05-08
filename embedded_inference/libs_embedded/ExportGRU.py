from .Quantizer import *


def _gru_export_weights(w_data_type, layer_id, weights_quant, bias_quant, postfix_name=""):
    var_weights = layer_id + "_weights" + postfix_name
    var_bias    = layer_id + "_bias" + postfix_name

    #weights
    code_weight = "const " + w_data_type + " " + var_weights + "[] = {" + "\n"
    for j in range(weights_quant.shape[0]):
        for i in range(weights_quant.shape[1]):
            code_weight+= str(weights_quant[j][i]) + ", " 
                
        code_weight+= "\n"

    code_weight+= "};\n\n"
    
    #bias
    code_weight+= "const " + w_data_type + " " + var_bias + "[] = {" + "\n"
    for i in range(bias_quant.shape[0]):
        code_weight+= str(bias_quant[i]) + ", " 
    code_weight+= "};\n\n"

    return code_weight, var_weights, var_bias

def _gru_add_padding(weights_raw, bias_raw, padding_inputs, padding_outputs):

    in_features  = weights_raw.shape[1]
    out_features = weights_raw.shape[0]

    p_out       = (padding_outputs - (out_features%padding_outputs))%padding_outputs
    p_in        = (padding_inputs  - (in_features%padding_inputs))%padding_inputs

    weights = numpy.zeros((out_features + p_out, in_features + p_in))
    weights[0:out_features, 0:in_features] = weights_raw

    bias        = numpy.zeros((out_features + p_out))
    bias[0:out_features] = bias_raw

    return weights, bias



def ExportGRU(layer, layer_num, network_prefix, input_shape, quantization_type):

    padding_inputs  = 4
    padding_outputs = 8

    w_hr, w_hz, w_hn = layer.weight_hh_l0.chunk(3, 0)
    w_ir, w_iz, w_in = layer.weight_ih_l0.chunk(3, 0)

    b_hr, b_hz, b_hn = layer.bias_hh_l0.chunk(3)
    b_ir, b_iz, b_in = layer.bias_ih_l0.chunk(3)


    w_hr = w_hr.to("cpu").detach().numpy()
    w_hz = w_hz.to("cpu").detach().numpy()
    w_hn = w_hn.to("cpu").detach().numpy()

    w_ir = w_ir.to("cpu").detach().numpy()
    w_iz = w_iz.to("cpu").detach().numpy()
    w_in = w_in.to("cpu").detach().numpy()

    b_hr = b_hr.to("cpu").detach().numpy()
    b_hz = b_hz.to("cpu").detach().numpy()
    b_hn = b_hn.to("cpu").detach().numpy()

    b_ir = b_ir.to("cpu").detach().numpy()
    b_iz = b_iz.to("cpu").detach().numpy()
    b_in = b_in.to("cpu").detach().numpy()
    
    #add padding
    w_hr, b_hr = _gru_add_padding(w_hr, b_hr, padding_inputs, padding_outputs)
    w_hz, b_hz = _gru_add_padding(w_hz, b_hz, padding_inputs, padding_outputs)
    w_hn, b_hn = _gru_add_padding(w_hn, b_hn, padding_inputs, padding_outputs)

    w_ir, b_ir = _gru_add_padding(w_ir, b_ir, padding_inputs, padding_outputs)
    w_iz, b_iz = _gru_add_padding(w_iz, b_iz, padding_inputs, padding_outputs)
    w_in, b_in = _gru_add_padding(w_in, b_in, padding_inputs, padding_outputs)


    layer_id = network_prefix + "_" + "layer_" + str(layer_num)
   
    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        w_hr_quant, b_hr_quant, hr_scale = Quantizer(w_hr, b_hr, max_value)
        w_hz_quant, b_hz_quant, hz_scale = Quantizer(w_hz, b_hz, max_value)
        w_hn_quant, b_hn_quant, hn_scale = Quantizer(w_hn, b_hn, max_value)

        w_ir_quant, b_ir_quant, ir_scale = Quantizer(w_ir, b_ir, max_value)
        w_iz_quant, b_iz_quant, iz_scale = Quantizer(w_iz, b_iz, max_value)
        w_in_quant, b_in_quant, in_scale = Quantizer(w_in, b_in, max_value)

        w_hr_quant      = numpy.round(w_hr_quant, 0).astype(int)
        b_hr_quant      = numpy.round(b_hr_quant, 0).astype(int)
        w_hz_quant      = numpy.round(w_hz_quant, 0).astype(int)
        b_hz_quant      = numpy.round(b_hz_quant, 0).astype(int)
        w_hn_quant      = numpy.round(w_hn_quant, 0).astype(int)
        b_hn_quant      = numpy.round(b_hn_quant, 0).astype(int)

        w_ir_quant      = numpy.round(w_ir_quant, 0).astype(int)
        b_ir_quant      = numpy.round(b_ir_quant, 0).astype(int)
        w_iz_quant      = numpy.round(w_iz_quant, 0).astype(int)
        b_iz_quant      = numpy.round(b_iz_quant, 0).astype(int)
        w_in_quant      = numpy.round(w_in_quant, 0).astype(int)
        b_in_quant      = numpy.round(b_in_quant, 0).astype(int)

    elif quantization_type == "int16": 
        io_data_type    = "int16_t"
        w_data_type     = "int16_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        w_hr_quant, b_hr_quant, hr_scale = Quantizer(w_hr, b_hr, max_value)
        w_hz_quant, b_hz_quant, hz_scale = Quantizer(w_hz, b_hz, max_value)
        w_hn_quant, b_hn_quant, hn_scale = Quantizer(w_hn, b_hn, max_value)

        w_ir_quant, b_ir_quant, ir_scale = Quantizer(w_ir, b_ir, max_value)
        w_iz_quant, b_iz_quant, iz_scale = Quantizer(w_iz, b_iz, max_value)
        w_in_quant, b_in_quant, in_scale = Quantizer(w_in, b_in, max_value)

        w_hr_quant      = numpy.round(w_hr_quant, 0).astype(int)
        b_hr_quant      = numpy.round(b_hr_quant, 0).astype(int)
        w_hz_quant      = numpy.round(w_hz_quant, 0).astype(int)
        b_hz_quant      = numpy.round(b_hz_quant, 0).astype(int)
        w_hn_quant      = numpy.round(w_hn_quant, 0).astype(int)
        b_hn_quant      = numpy.round(b_hn_quant, 0).astype(int)

        w_ir_quant      = numpy.round(w_ir_quant, 0).astype(int)
        b_ir_quant      = numpy.round(b_ir_quant, 0).astype(int)
        w_iz_quant      = numpy.round(w_iz_quant, 0).astype(int)
        b_iz_quant      = numpy.round(b_iz_quant, 0).astype(int)
        w_in_quant      = numpy.round(w_in_quant, 0).astype(int)
        b_in_quant      = numpy.round(b_in_quant, 0).astype(int)

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

        hr_scale        = 1024
        hz_scale        = 1024
        hn_scale        = 1024
        ir_scale        = 1024
        iz_scale        = 1024
        in_scale        = 1024

        w_hr_quant, b_hr_quant = w_hr, b_hr
        w_hz_quant, b_hz_quant = w_hz, b_hz
        w_hn_quant, b_hn_quant = w_hn, b_hn

        w_ir_quant, b_ir_quant = w_ir, b_ir
        w_iz_quant, b_iz_quant = w_iz, b_iz
        w_in_quant, b_in_quant = w_in, b_in


    input_size          = w_ir_quant.shape[1]
    sequence_length     = input_shape[1]
    output_size         = w_ir_quant.shape[0]

    '''
    print("ExportGRU")
    print(input_shape)
    print(w_hr_quant.shape, b_hr_quant.shape)
    print(w_hz_quant.shape, b_hz_quant.shape)
    print(w_hn_quant.shape, b_hn_quant.shape)
    print(w_ir_quant.shape, b_ir_quant.shape)
    print(w_iz_quant.shape, b_iz_quant.shape)
    print(w_in_quant.shape, b_in_quant.shape)
    print(input_size, output_size)
    print("\n\n\n")
    '''
    
    wb_hr_code, var_w_hr, var_b_hr = _gru_export_weights(w_data_type, layer_id, w_hr_quant, b_hr_quant, "_hr")
    wb_hz_code, var_w_hz, var_b_hz = _gru_export_weights(w_data_type, layer_id, w_hz_quant, b_hz_quant, "_hz")
    wb_hn_code, var_w_hn, var_b_hn = _gru_export_weights(w_data_type, layer_id, w_hn_quant, b_hn_quant, "_hn")

    wb_ir_code, var_w_ir, var_b_ir = _gru_export_weights(w_data_type, layer_id, w_ir_quant, b_ir_quant, "_ir")
    wb_iz_code, var_w_iz, var_b_iz = _gru_export_weights(w_data_type, layer_id, w_iz_quant, b_iz_quant, "_iz")
    wb_in_code, var_w_in, var_b_in = _gru_export_weights(w_data_type, layer_id, w_in_quant, b_in_quant, "_in")

    code_weight = wb_hr_code + wb_hz_code + wb_hn_code + wb_ir_code + wb_iz_code + wb_in_code + "\n\n"

    code_network = "" 

    #layer call code
    code_network+= "\tGRU<" + str(input_size) + ", " + str(output_size) + ", " 
    code_network+= io_data_type + ", " + w_data_type + ", " + acc_data_type + ", "
    code_network+= str(max_value) + ", " 
    code_network+= str(hr_scale) + ", "
    code_network+= str(hz_scale) + ", "
    code_network+= str(hn_scale) + ", "
    code_network+= str(ir_scale) + ", "
    code_network+= str(iz_scale) + ", "
    code_network+= str(in_scale)
    code_network+= ">"

    code_network+= "(\n\t\toutput_buffer(), input_buffer(),\n"
    code_network+= "\t\t" + str(sequence_length) + ",\n"
    code_network+= "\t\t" + var_w_hr + ", " + var_b_hr + ",\n"
    code_network+= "\t\t" + var_w_hz + ", " + var_b_hz + ",\n"
    code_network+= "\t\t" + var_w_hn + ", " + var_b_hn + ",\n"
    code_network+= "\t\t" + var_w_ir + ", " + var_b_ir + ",\n"
    code_network+= "\t\t" + var_w_iz + ", " + var_b_iz + ",\n"
    code_network+= "\t\t" + var_w_in + ", " + var_b_in + ");\n"

    code_network+= "\tswap_buffer();" + "\n\n"


    code = (code_network, code_weight)
    macs = sequence_length*3*output_size*(output_size + input_size + 1 + 4)

    print("export_GRU :")
    print("quantization         ", quantization_type)
    print("output_size          ", output_size)
    print("input_size           ", input_size)
    print("sequence_length      ", sequence_length)
    print("macs                 ", macs)
    print("\n\n")

    return code, (output_size, ), output_size, macs




def ExportGRUStream(layer, layer_num, network_prefix, input_shape, quantization_type):

    padding_inputs  = 4
    padding_outputs = 8

    w_hr, w_hz, w_hn = layer.weight_hh_l0.chunk(3, 0)
    w_ir, w_iz, w_in = layer.weight_ih_l0.chunk(3, 0)

    b_hr, b_hz, b_hn = layer.bias_hh_l0.chunk(3)
    b_ir, b_iz, b_in = layer.bias_ih_l0.chunk(3)


    w_hr = w_hr.to("cpu").detach().numpy()
    w_hz = w_hz.to("cpu").detach().numpy()
    w_hn = w_hn.to("cpu").detach().numpy()

    w_ir = w_ir.to("cpu").detach().numpy()
    w_iz = w_iz.to("cpu").detach().numpy()
    w_in = w_in.to("cpu").detach().numpy()

    b_hr = b_hr.to("cpu").detach().numpy()
    b_hz = b_hz.to("cpu").detach().numpy()
    b_hn = b_hn.to("cpu").detach().numpy()

    b_ir = b_ir.to("cpu").detach().numpy()
    b_iz = b_iz.to("cpu").detach().numpy()
    b_in = b_in.to("cpu").detach().numpy()
    
    #add padding
    w_hr, b_hr = _gru_add_padding(w_hr, b_hr, padding_inputs, padding_outputs)
    w_hz, b_hz = _gru_add_padding(w_hz, b_hz, padding_inputs, padding_outputs)
    w_hn, b_hn = _gru_add_padding(w_hn, b_hn, padding_inputs, padding_outputs)

    w_ir, b_ir = _gru_add_padding(w_ir, b_ir, padding_inputs, padding_outputs)
    w_iz, b_iz = _gru_add_padding(w_iz, b_iz, padding_inputs, padding_outputs)
    w_in, b_in = _gru_add_padding(w_in, b_in, padding_inputs, padding_outputs)


    layer_id = network_prefix + "_" + "layer_" + str(layer_num)
   
    if quantization_type == "int8":
        io_data_type    = "int8_t"
        w_data_type     = "int8_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        w_hr_quant, b_hr_quant, hr_scale = Quantizer(w_hr, b_hr, max_value)
        w_hz_quant, b_hz_quant, hz_scale = Quantizer(w_hz, b_hz, max_value)
        w_hn_quant, b_hn_quant, hn_scale = Quantizer(w_hn, b_hn, max_value)

        w_ir_quant, b_ir_quant, ir_scale = Quantizer(w_ir, b_ir, max_value)
        w_iz_quant, b_iz_quant, iz_scale = Quantizer(w_iz, b_iz, max_value)
        w_in_quant, b_in_quant, in_scale = Quantizer(w_in, b_in, max_value)

        w_hr_quant      = numpy.round(w_hr_quant, 0).astype(int)
        b_hr_quant      = numpy.round(b_hr_quant, 0).astype(int)
        w_hz_quant      = numpy.round(w_hz_quant, 0).astype(int)
        b_hz_quant      = numpy.round(b_hz_quant, 0).astype(int)
        w_hn_quant      = numpy.round(w_hn_quant, 0).astype(int)
        b_hn_quant      = numpy.round(b_hn_quant, 0).astype(int)

        w_ir_quant      = numpy.round(w_ir_quant, 0).astype(int)
        b_ir_quant      = numpy.round(b_ir_quant, 0).astype(int)
        w_iz_quant      = numpy.round(w_iz_quant, 0).astype(int)
        b_iz_quant      = numpy.round(b_iz_quant, 0).astype(int)
        w_in_quant      = numpy.round(w_in_quant, 0).astype(int)
        b_in_quant      = numpy.round(b_in_quant, 0).astype(int)

    elif quantization_type == "int16": 
        io_data_type    = "int16_t"
        w_data_type     = "int16_t"
        acc_data_type   = "int32_t"
        max_value       = 128-1

        w_hr_quant, b_hr_quant, hr_scale = Quantizer(w_hr, b_hr, max_value)
        w_hz_quant, b_hz_quant, hz_scale = Quantizer(w_hz, b_hz, max_value)
        w_hn_quant, b_hn_quant, hn_scale = Quantizer(w_hn, b_hn, max_value)

        w_ir_quant, b_ir_quant, ir_scale = Quantizer(w_ir, b_ir, max_value)
        w_iz_quant, b_iz_quant, iz_scale = Quantizer(w_iz, b_iz, max_value)
        w_in_quant, b_in_quant, in_scale = Quantizer(w_in, b_in, max_value)

        w_hr_quant      = numpy.round(w_hr_quant, 0).astype(int)
        b_hr_quant      = numpy.round(b_hr_quant, 0).astype(int)
        w_hz_quant      = numpy.round(w_hz_quant, 0).astype(int)
        b_hz_quant      = numpy.round(b_hz_quant, 0).astype(int)
        w_hn_quant      = numpy.round(w_hn_quant, 0).astype(int)
        b_hn_quant      = numpy.round(b_hn_quant, 0).astype(int)

        w_ir_quant      = numpy.round(w_ir_quant, 0).astype(int)
        b_ir_quant      = numpy.round(b_ir_quant, 0).astype(int)
        w_iz_quant      = numpy.round(w_iz_quant, 0).astype(int)
        b_iz_quant      = numpy.round(b_iz_quant, 0).astype(int)
        w_in_quant      = numpy.round(w_in_quant, 0).astype(int)
        b_in_quant      = numpy.round(b_in_quant, 0).astype(int)

    else:
        io_data_type    = "float"
        w_data_type     = "float"
        acc_data_type   = "float"
        max_value       = 0

        hr_scale        = 1024
        hz_scale        = 1024
        hn_scale        = 1024
        ir_scale        = 1024
        iz_scale        = 1024
        in_scale        = 1024

        w_hr_quant, b_hr_quant = w_hr, b_hr
        w_hz_quant, b_hz_quant = w_hz, b_hz
        w_hn_quant, b_hn_quant = w_hn, b_hn

        w_ir_quant, b_ir_quant = w_ir, b_ir
        w_iz_quant, b_iz_quant = w_iz, b_iz
        w_in_quant, b_in_quant = w_in, b_in


    input_size          = w_ir_quant.shape[1]
    sequence_length     = input_shape[1]
    output_size         = w_ir_quant.shape[0]

    '''
    print("ExportGRU")
    print(input_shape)
    print(w_hr_quant.shape, b_hr_quant.shape)
    print(w_hz_quant.shape, b_hz_quant.shape)
    print(w_hn_quant.shape, b_hn_quant.shape)
    print(w_ir_quant.shape, b_ir_quant.shape)
    print(w_iz_quant.shape, b_iz_quant.shape)
    print(w_in_quant.shape, b_in_quant.shape)
    print(input_size, output_size)
    print("\n\n\n")
    '''
    
    wb_hr_code, var_w_hr, var_b_hr = _gru_export_weights(w_data_type, layer_id, w_hr_quant, b_hr_quant, "_hr")
    wb_hz_code, var_w_hz, var_b_hz = _gru_export_weights(w_data_type, layer_id, w_hz_quant, b_hz_quant, "_hz")
    wb_hn_code, var_w_hn, var_b_hn = _gru_export_weights(w_data_type, layer_id, w_hn_quant, b_hn_quant, "_hn")

    wb_ir_code, var_w_ir, var_b_ir = _gru_export_weights(w_data_type, layer_id, w_ir_quant, b_ir_quant, "_ir")
    wb_iz_code, var_w_iz, var_b_iz = _gru_export_weights(w_data_type, layer_id, w_iz_quant, b_iz_quant, "_iz")
    wb_in_code, var_w_in, var_b_in = _gru_export_weights(w_data_type, layer_id, w_in_quant, b_in_quant, "_in")

    code_weight = wb_hr_code + wb_hz_code + wb_hn_code + wb_ir_code + wb_iz_code + wb_in_code + "\n\n"

    code_network = "" 

    #layer call code
    code_network+= "\tGRUStream<" + str(input_size) + ", " + str(output_size) + ", " 
    code_network+= io_data_type + ", " + w_data_type + ", " + acc_data_type + ", "
    code_network+= str(max_value) + ", " 
    code_network+= str(hr_scale) + ", "
    code_network+= str(hz_scale) + ", "
    code_network+= str(hn_scale) + ", "
    code_network+= str(ir_scale) + ", "
    code_network+= str(iz_scale) + ", "
    code_network+= str(in_scale)
    code_network+= ">"

    code_network+= "(\n\t\toutput_buffer(), input_buffer(), hidden_state, \n"
    code_network+= "\t\t" + var_w_hr + ", " + var_b_hr + ",\n"
    code_network+= "\t\t" + var_w_hz + ", " + var_b_hz + ",\n"
    code_network+= "\t\t" + var_w_hn + ", " + var_b_hn + ",\n"
    code_network+= "\t\t" + var_w_ir + ", " + var_b_ir + ",\n"
    code_network+= "\t\t" + var_w_iz + ", " + var_b_iz + ",\n"
    code_network+= "\t\t" + var_w_in + ", " + var_b_in + ");\n"

    code_network+= "\tswap_buffer();" + "\n\n"


    code = (code_network, code_weight)
    macs = 3*output_size*(output_size + input_size + 1 + 4)

    print("export_GRU :")
    print("quantization         ", quantization_type)
    print("output_size          ", output_size)
    print("input_size           ", input_size)
    print("sequence_length      ", sequence_length)
    print("macs                 ", macs)
    print("\n\n")

    return code, (output_size, ), output_size, macs, output_size