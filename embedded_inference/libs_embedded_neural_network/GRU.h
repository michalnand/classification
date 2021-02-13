#ifndef _GRU_H_
#define _GRU_H_

#include <stdint.h>
#include <typeinfo>

#include "dot_microkernel.h"


#include <math.h>


template<class DType>
float _sigmoid(float x) 
{
    if (typeid(DType) == typeid(float)) 
        return 1.0/(1.0 + exp(-x));  
    else  
        return 127.0/(1.0 + exp(-x/127.0));  
}

template<class DType>
float _tanh(float x) 
{
    if (typeid(DType) == typeid(float)) 
        return tanh(x);    
    else
        return 127.0*tanh(x/127.0);    
}



template<   unsigned int in_features, unsigned int hidden_size, 
            class IO_t, class WEIGHT_t, class ACC_t, int io_max, 
            int hr_scale, int hz_scale, int hn_scale,
            int ir_scale, int iz_scale, int in_scale>
void GRUStep(float *hidden_buffer_next, float *hidden_buffer, IO_t *input_buffer,
             
            const WEIGHT_t *whr, const WEIGHT_t *bhr,
            const WEIGHT_t *whz, const WEIGHT_t *bhz,
            const WEIGHT_t *whn, const WEIGHT_t *bhn,

            const WEIGHT_t *wir, const WEIGHT_t *bir,
            const WEIGHT_t *wiz, const WEIGHT_t *biz,
            const WEIGHT_t *win, const WEIGHT_t *bin )
{
    IO_t hidden_buffer_quant[hidden_size];

    for (unsigned int i = 0; i < hidden_size; i++)
    {
        if (typeid(IO_t) == typeid(float))
        { 
            hidden_buffer_quant[i] = hidden_buffer[i];
        }
        else
        {
            hidden_buffer_quant[i] = (IO_t)(hidden_buffer[i]*127.0);
        }
    } 

    for (unsigned int i = 0; i < hidden_size; i++)
    {       
        float ra  = dot_microkernel<hidden_size, WEIGHT_t, IO_t, ACC_t>(whr + i*hidden_size, hidden_buffer_quant);
        if (typeid(IO_t) != typeid(float))
            ra  = (ra*hr_scale)/(128*128*1024.0) + (bhr[i]*hr_scale)/(128*1024.0);
        else
            ra  = ra + bhr[i];

        float rb  = dot_microkernel<in_features, WEIGHT_t, IO_t, ACC_t>(wir + i*in_features, input_buffer);
        if (typeid(IO_t) != typeid(float))
            rb  = (rb*hr_scale)/(128*128*1024.0) + (bir[i]*hr_scale)/(128*1024.0);
        else
            rb  = rb + bir[i];

        float r = _sigmoid<float>(ra + rb);

        float za  = dot_microkernel<hidden_size, WEIGHT_t, IO_t, ACC_t>(whz + i*hidden_size, hidden_buffer_quant);
        if (typeid(IO_t) != typeid(float))
            za  = (za*hz_scale)/(128*128*1024.0) + (bhz[i]*hz_scale)/(128*1024.0);
        else
            za  = za + bhz[i];

        float zb  = dot_microkernel<in_features, WEIGHT_t, IO_t, ACC_t>(wiz + i*in_features, input_buffer);
        if (typeid(IO_t) != typeid(float))
            zb  = (zb*iz_scale)/(128*128*1024.0) + (biz[i]*iz_scale)/(128*1024.0);
        else
            zb  = zb + biz[i];

        float z = _sigmoid<float>(za + zb);
 
        float na  = dot_microkernel<hidden_size, WEIGHT_t, IO_t, ACC_t>(whn + i*hidden_size, hidden_buffer_quant);
        if (typeid(IO_t) != typeid(float))
            na  = (na*hn_scale)/(128*128*1024.0) + (bhn[i]*hn_scale)/(128*1024.0);
        else
            na  = na + bhn[i];

        float nb  = dot_microkernel<in_features, WEIGHT_t, IO_t, ACC_t>(win + i*in_features, input_buffer);
        if (typeid(IO_t) != typeid(float))
            nb  = (nb*in_scale)/(128*128*1024.0) + (bin[i]*in_scale)/(128*1024.0);
        else 
            nb  = nb + bin[i];

        float n = _tanh<float>(r*na + nb); 

        hidden_buffer_next[i] = (1.0 - z)*n + z*hidden_buffer[i];
    }
}

template<   unsigned int in_features, unsigned int hidden_size, 
            class IO_t, class WEIGHT_t, class ACC_t, int io_max, 
            int hr_scale, int hz_scale, int hn_scale,
            int ir_scale, int iz_scale, int in_scale>
void GRU(   IO_t *output_buffer, IO_t *input_buffer, unsigned int sequence_length,
            
            const WEIGHT_t *whr, const WEIGHT_t *bhr,
            const WEIGHT_t *whz, const WEIGHT_t *bhz,
            const WEIGHT_t *whn, const WEIGHT_t *bhn,

            const WEIGHT_t *wir, const WEIGHT_t *bir,
            const WEIGHT_t *wiz, const WEIGHT_t *biz,
            const WEIGHT_t *win, const WEIGHT_t *bin )
{ 
    float hidden_buffer[hidden_size];
    float hidden_buffer_next[hidden_size];

    for (unsigned int i = 0; i < hidden_size; i++)
    {
        hidden_buffer[i]       = 0;
        hidden_buffer_next[i]  = 0;
    }

    for (unsigned int t = 0; t < sequence_length; t++)
    {    
        GRUStep<in_features, hidden_size,  IO_t, WEIGHT_t, ACC_t, io_max, 
                hr_scale,   hz_scale,     hn_scale,
                ir_scale,   iz_scale,     in_scale>(   
                    hidden_buffer_next, hidden_buffer, input_buffer + t*in_features,
                    whr, bhr,
                    whz, bhz,
                    whn, bhn,

                    wir, bir,
                    wiz, biz,
                    win, bin 
        ); 

        for (unsigned int i = 0; i < hidden_size; i++)
            hidden_buffer[i] = hidden_buffer_next[i];
    }
   
    for (unsigned int i = 0; i < hidden_size; i++)
    {
        if (typeid(IO_t) == typeid(float))
        {
            output_buffer[i] = hidden_buffer[i];
        }
        else
        {
            output_buffer[i] = (IO_t)(hidden_buffer[i]*127.0);
        }
    }
}


#endif