/*
 * norm.h
 *
 *  Created on: Jan 26, 2018
 *      Author: byrdie
 */

#ifndef NORM_H_
#define NORM_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>

namespace kso {

namespace img {

namespace dspk {

__global__ void calc_norm_0(float * norm_0, float * gm, uint * bad_pix, dim3 sz, uint k_sz);
__global__ void calc_norm_1(float * norm_1, float * norm_0, dim3 sz, uint k_sz);
__global__ void calc_norm_2(float * norm_2, float * norm_1, dim3 sz, uint k_sz);

}

}

}


#endif /* NORM_H_ */
