/*
 * gm.h
 *
 *  Created on: Jan 26, 2018
 *      Author: byrdie
 */

#ifndef GM_H_
#define GM_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>

namespace kso {

namespace img {

namespace dspk {

__global__ void calc_gm(float * gm, float * gdev, float * nsd, float std_dev, uint * new_bad, dim3 sz, uint k_sz);
__global__ void init_gm(float* gm, dim3 sz);


}

}

}



#endif /* GM_H_ */
