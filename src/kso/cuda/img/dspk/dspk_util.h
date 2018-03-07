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
	uint ndim, nmet;
	float mem_fill;
	dim3 sz, csz;	// shape of original data and chunked data
	dim3 st, cst;	// array stride of original data and chunked data'
	dim3 hsz, hst;	// size and stride of histogram
	dim3 sb;		// array stride in bytes
	uint sz3, csz3, hsz3;
	uint ksz, ks2;	// kernel size, kernel half-size
	float * dt, * gm;		// host memory
	float * q2;
	float * buf_d;
	float * dt_d, * gm_d;
	float * q2_d;
	float * gdev_d, *nsd_d, *tmp_d, *norm_d;	// device memory
	uint *newBad, *newBad_d;				// more device memory
	float * ht, * ht_d;		// histogram memory
	float * cs, * cs_d;		// cumsum memory
	float * t0, * t1, * t0_d, * t1_d;		// smoothed arrays of thresholds for each median
	float * T0_d, * T1_d;	// unsmoothed thresholds


	dim3 threads;		// number of threads per block
	dim3 blocks;	// number of blocks

	kso::util::stride * S;

	buf(float * data, float * goodmap, dim3 data_sz, uint kern_sz, dim3 hist_sz, uint n_threads);
	buf(std::string path, uint max_sz, uint kern_sz, dim3 hist_sz, uint n_threads);
	void prep(uint kern_sz, dim3 hist_sz, uint n_threads);


};

}

}

}


#endif /* DSPK_UTIL_H_ */
