/*
 * gdev.h
 *
 *  Created on: Jan 26, 2018
 *      Author: byrdie
 */

#ifndef GDEV_H_
#define GDEV_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>


namespace kso {

namespace img {

namespace dspk {

__global__ void calc_gdev_0(float * gdev_0, float * dt, float * gm, dim3 sz, uint k_sz);
__global__ void calc_gdev_1(float * gdev_1, float * gdev_0, dim3 sz, uint k_sz);
__global__ void calc_gdev_2(float * gdev_2, float * gdev_1, float * dt, float * gm, float * norm, dim3 sz, uint k_sz);


}

}

}


#endif /* GDEV_H_ */
