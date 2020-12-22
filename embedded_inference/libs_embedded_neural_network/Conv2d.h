#ifndef _Conv2d_H_
#define _Conv2d_H_

#include <stdint.h>

#include "dot_microkernel.h"

template<   unsigned int height, unsigned int width, 
            unsigned int input_channels, unsigned int output_channels, 
            unsigned int kernel_size, unsigned int stride,
            class IO_t, class WEIGHT_t, class ACC_t, int io_max, int weight_max>
void Conv2d(IO_t *output_buffer, IO_t *input_buffer, const WEIGHT_t *kernel, const WEIGHT_t *bias, int scale)
{
    auto output_height  = (height - (kernel_size - 1) - 1)/stride + 1;
    auto output_width   = (width  - (kernel_size - 1) - 1)/stride + 1;

    for (unsigned int filter = 0; filter < output_channels; filter++)
        for (unsigned int y = 0; y < output_height; y++)
            for (unsigned int x = 0; x < output_width; x++)
            {
                const WEIGHT_t *kernel_ =  &(kernel[filter*kernel_size*kernel_size*input_channels]);

                ACC_t result = 0;

                for (unsigned int ky = 0; ky < kernel_size; ky++)
                {  
                    unsigned int input_idx = (y*stride*width + ky*width + x*stride)*input_channels;
                    IO_t *input_buffer_ = &(input_buffer[input_idx]);
                         
                    result+= dot_microkernel<kernel_size*input_channels, IO_t, WEIGHT_t, ACC_t>(input_buffer_, kernel_);
 
                    kernel_+= kernel_size*input_channels;
                }
                
                result = ((result + bias[filter])*scale)/(1024*weight_max);
               
                if (io_max != 1)
                {
                    if (result > io_max) 
                        result = io_max;
                    
                    if (result < -io_max)
                        result = -io_max;
                }
  
                output_buffer[(y*output_width + x)*output_channels + filter]   = result; 
            }
}




#endif