/*
 * dspk_cuda.h
 *
 *  Created on: Jan 22, 2018
 *      Author: byrdie
 */

#ifndef DSPK_CUDA_H_
#define DSPK_CUDA_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

//test
namespace p = boost::python;
namespace np = boost::python::numpy;

namespace kso {

namespace img {

namespace dspk {

np::ndarray locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter);
__global__ void calc_dev(float * dt, float * gm, float * dev, dim3 sz, uint k_sz);
__global__ void calc_goodmap(float std_dev, float * gm, float * dev, dim3 sz, uint k_sz, uint * new_bad);

}

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

#endif /* DSPK_CUDA_H_ */
