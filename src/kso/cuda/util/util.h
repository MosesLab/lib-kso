/*
 * util.h
 *
 *  Created on: Jan 26, 2018
 *      Author: byrdie
 */

#ifndef UTIL_H_
#define UTIL_H_


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>



namespace kso {

namespace util {


void enum_device();
size_t get_device_mem(uint device);
dim3 add_dim3(dim3 a, dim3 b);


}

}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}



#endif /* UTIL_H_ */
