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

#include "util/util.h"
#include "util/stride.h"

#include "instrument/IRIS/read_fits.h"


namespace kso {

namespace img {

namespace dspk {

class buf {
public:
	dim3 sz, csz;	// shape of original data and chunked data
	dim3 st, cst;	// array stride of original data and chunked data'
	dim3 sb;		// array stride in bytes
	uint sz3, csz3;
	uint ksz, ks2;	// kernel size, kernel half-size
	float * dt, * gm;		// host memory
	float * q1, * q2, * q3;
	float * dt_d, * gm_d, * gdev_d, *nsd_d, *tmp_d, *norm_d;	// device memory
	uint *newBad, *newBad_d;				// more device memory


	dim3 threads;		// number of threads per block
	dim3 blocks;	// number of blocks

	kso::util::stride * S;

	buf(float * data, float * goodmap, dim3 data_sz, uint kern_sz, uint n_threads);
	buf(std::string path, uint max_sz, uint kern_sz, uint n_threads);


};

}

}

}


#endif /* DSPK_UTIL_H_ */
