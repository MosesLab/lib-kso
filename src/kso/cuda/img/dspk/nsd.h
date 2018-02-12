/*
 * nsd.h
 *
 *  Created on: Jan 26, 2018
 *      Author: byrdie
 */

#ifndef NSD_H_
#define NSD_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>

#include "gdev.h"

namespace kso {

namespace img {

namespace dspk {

__device__ float nsd_kern_1D(float X, float ks2, float sig);

__global__ void calc_nsd_0(float * nsd_0, float * gdev, dim3 sz, uint k_sz);
__global__ void calc_nsd_1(float * nsd_1, float * nsd_0, dim3 sz, uint k_sz);
__global__ void calc_nsd_2(float * nsd_2, float * nsd_1, float * norm, dim3 sz, uint k_sz);


}

}

}


#endif /* NSD_H_ */
