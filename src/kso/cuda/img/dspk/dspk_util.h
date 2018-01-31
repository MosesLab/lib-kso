/*
 * dspk_util.h
 *
 *  Created on: Jan 30, 2018
 *      Author: byrdie
 */

#ifndef DSPK_UTIL_H_
#define DSPK_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cuda.h>

#include "util/stride.h"



namespace kso {

namespace img {

namespace dspk {

class buf {
public:
	dim3 sz, csz;
	uint sz3, csz3;
	uint ksz;	// kernel size
	float * dt, * gm;		// host memory
	float * dt_d, * gm_d, * gdev_d, *nsd_d, *tmp_d, *norm_d;	// device memory
	uint * newBad_d;				// more device memory

	kso::util::stride S;

	buf(float * data, dim3 data_sz, uint kern_sz, uint n_threads);


};

}

}

}


#endif /* DSPK_UTIL_H_ */
