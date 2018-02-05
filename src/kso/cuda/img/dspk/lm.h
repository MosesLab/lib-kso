/*
 * lm.h
 *
 *  Created on: Feb 3, 2018
 *      Author: byrdie
 */

#ifndef LM_H_
#define LM_H_



namespace kso {

namespace img {

namespace dspk {




float local_kern_1D(dim3 X, uint ksz);

__global__ void calc_dt_0(float * dt_0, float * dt, float * gm, dim3 sz, uint k_sz);
__global__ void calc_dt_1(float * dt_1, float * dt_0, dim3 sz, uint k_sz);
__global__ void calc_dt_2(float * dt_2, float * dt_1, float * gm, float * norm, dim3 sz, uint k_sz);

}

}

}

#endif /* LM_H_ */
