#ifndef _Conv1d_H_
#define _Conv1d_H_

#include <stdint.h>

#include "dot_microkernel.h"

template<   unsigned int width, 
            unsigned int input_channels, unsigned int output_channels, 
            unsigned int kernel_size, unsigned int stride,
            class IO_t, class WEIGHT_t, class ACC_t, int io_max, int zero_point, int scale>
void Conv1d(IO_t *output_buffer, IO_t *input_buffer, const WEIGHT_t *kernel, const WEIGHT_t *bias)
{
    auto output_width  = (width - (kernel_size - 1) - 1)/stride + 1;
    
    for (unsigned int filter = 0; filter < output_channels; filter++)
        for (unsigned int x = 0; x < output_width; x++)
        {
            const WEIGHT_t *kernel_ =  &(kernel[filter*kernel_size*input_channels]);

            IO_t *input_buffer_     = &(input_buffer[x*stride*input_channels]); 
                
            ACC_t result = dot_microkernel<kernel_size*input_channels, IO_t, WEIGHT_t, ACC_t>(input_buffer_, kernel_);

            result = ((result + bias[filter])*scale + zero_point)/256;
            
            if (io_max != 0)
            {
                if (result > io_max) 
                    result = io_max;
                    
                if (result < -io_max)
                    result = -io_max;
            }
           
            output_buffer[x*output_channels + filter]   = result;
        }
}



#endif