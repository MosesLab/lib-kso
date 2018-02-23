/*
 * med.h
 *
 *  Created on: Feb 13, 2018
 *      Author: byrdie
 */

#ifndef MED_H_
#define MED_H_

#include <vector>

#include "util/util.h"

namespace kso {

namespace img {

namespace dspk {

// store domain of histogram
const int dt_max = 16384;
const int dt_min = -200;
const int Dt = dt_max - dt_min;

__global__ void calc_gm(float * gm, uint * new_bad, float * dt, float * q2, float * t0, float * t1, dim3 sz, dim3 hsz, uint ndim);

__global__ void calc_thresh(float * t0, float * t1, float * hist, float * cs, dim3 hsz, float T0, float T1);

__global__ void calc_cumsum(float * cs, float * hist, dim3 hsz);

__global__ void calc_hist(float * hist, float * dt, float * q2, dim3 sz, dim3 hsz);

void calc_quartile(float * q, float * dt, float * gm, float * tmp, dim3 sz, dim3 ksz, uint quartile);

void calc_quartiles(float * q1, float * q2, float * q3, float * dt, float * gm, float * tmp, dim3 sz, dim3 ksz);

__global__ void calc_sep_quartile(float * q_out, float * q_in, float * gm, dim3 sz, dim3 ksz, dim3 axis, uint quartile);
__global__ void calc_tot_quartile(float * q, float * dt, float * gm, dim3 sz, dim3 ksz, uint quartile);

__global__ void init_hist(float * hist, float * t0, float * t1, dim3 hsz, uint ndim);

//__global__ void calc_median_hist(float * hist, float * q2, float * dt, float * gm, dim3 sz, dim3 ksz, )

}

}

}



#endif /* MED_H_ */
