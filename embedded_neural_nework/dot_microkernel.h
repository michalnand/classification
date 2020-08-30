#ifndef _dot_microkernel_H_
#define _dot_microkernel_H_



template<unsigned int vector_size, class DATA_VA_t, class DATA_VB_t, class ACC_t>
ACC_t dot_microkernel(const DATA_VA_t *va, const DATA_VB_t *vb)
{
    ACC_t buffer[32];
    for (unsigned int i = 0; i < 32; i++)
        buffer[i] = 0;

    unsigned int size = vector_size;
    
    unsigned int idx  = 0; 

    while (size >= 32)
    {
        buffer[0]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[1]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[2]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[3]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[4]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[5]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[6]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[7]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;

        buffer[8]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[9]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[10]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[11]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[12]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[13]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[14]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[15]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;

        buffer[16]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[17]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[18]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[19]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[20]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[21]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[22]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[23]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;

        buffer[24]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[25]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[26]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[27]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[28]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[29]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[30]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[31]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
  
        size-= 32; 
    }

    while (size >= 16)
    {
        buffer[0]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[1]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[2]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[3]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[4]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[5]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[6]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[7]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;

        buffer[8]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[9]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[10]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[11]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[12]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[13]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[14]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[15]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;

        size-= 8;
    }

    while (size >= 8)
    {
        buffer[0]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[1]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[2]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[3]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[4]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[5]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[6]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
        buffer[7]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;
  
        size-= 8;
    }


    while (size != 0)
    {
        buffer[0]+= (ACC_t)va[idx]*(ACC_t)vb[idx]; idx++;  
        size--;
    }


    ACC_t result = 0;
    for (unsigned int i = 0; i < 32; i++)
        result+= buffer[i];

    return result;
}


#endif