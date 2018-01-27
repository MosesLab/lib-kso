/*
 * dspk_cuda.h
 *
 *  Created on: Jan 22, 2018
 *      Author: byrdie
 */

#ifndef DSPK_H_
#define DSPK_H_

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
__global__ void calc_nm_x(float * dt, float * nm_out,  float * gm, float * nrm_out, dim3 sz, uint k_sz);
__global__ void calc_nm_y(float * nm_in, float * nm_out, float * nrm_in, float * nrm_out, dim3 sz, uint k_sz);
__global__ void calc_nm_z(float * nm_in, float * nm_out, float * nrm_in, float * nrm_out, dim3 sz, uint k_sz);

__global__ void calc_dev(float * dt, float * nm, float * dev, dim3 sz);

__global__ void calc_nsd_x(float * dev, float * nsd_out, float * gm, dim3 sz, uint k_sz);
__global__ void calc_nsd_y(float * nsd_in, float * nsd_out, dim3 sz, uint k_sz);
__global__ void calc_nsd_z(float * nsd_in, float * nsd_out, float * nrm, dim3 sz, uint k_sz);

__global__ void update_gm(float std_dev, float * gm, float * dev, float * nsd, dim3 sz, uint * new_bad);


__global__ void calc_norm_0(float * norm_0, float * gm, dim3 sz, uint k_sz);
__global__ void calc_norm_1(float * norm_1, float * norm_0, dim3 sz, uint k_sz);
__global__ void calc_norm_2(float * norm_2, float * norm_1, dim3 sz, uint k_sz);

__global__ void calc_gdev_0(float * gdev_0, float * dt, float * gm, dim3 sz, uint k_sz);
__global__ void calc_gdev_1(float * gdev_1, float * gdev_0, dim3 sz, uint k_sz);
__global__ void calc_gdev_2(float * gdev_2, float * gdev_1, float * dt, float * gm, float * norm, dim3 sz, uint k_sz);

__global__ void calc_nsd_0(float * nsd_0, float * gdev, dim3 sz, uint k_sz);
__global__ void calc_nsd_1(float * nsd_1, float * nsd_0, dim3 sz, uint k_sz);
__global__ void calc_nsd_2(float * nsd_2, float * nsd_1, float * norm, dim3 sz, uint k_sz);

__global__ void calc_gm(float * gm, float * gdev, float * nsd, float std_dev, uint * new_bad, dim3 sz, uint k_sz);




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
