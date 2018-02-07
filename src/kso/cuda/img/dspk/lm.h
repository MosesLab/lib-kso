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




__device__ float local_kern_1D(uint X, uint ksz);

__global__ void calc_dt_0(float * dt_0, float * dt, float * gm, dim3 sz, uint k_sz);
__global__ void calc_dt_1(float * dt_1, float * dt_0, dim3 sz, uint k_sz);
__global__ void calc_dt_2(float * dt_2, float * dt_1, float * gm, float * norm, dim3 sz, uint k_sz);

__global__ void calc_lmn_0(float * norm_0, float * gm, uint * bad_pix, dim3 sz, uint k_sz);
__global__ void calc_lmn_1(float * norm_1, float * norm_0, dim3 sz, uint k_sz);
__global__ void calc_lmn_2(float * norm_2, float * norm_1, dim3 sz, uint k_sz);

}

}

}

#endif /* LM_H_ */
