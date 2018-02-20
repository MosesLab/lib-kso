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

void calc_quartiles(float * q1, float * q2, float * q3, float * dt, float * gm, float * tmp, dim3 sz, dim3 ksz);

__global__ void calc_sep_quartile(float * q_out, float * q_in, float * gm, dim3 sz, dim3 ksz, dim3 axis, uint quartile);

//__global__ void calc_median_hist(float * hist, float * q2, float * dt, float * gm, dim3 sz, dim3 ksz, )

}

}

}



#endif /* MED_H_ */
