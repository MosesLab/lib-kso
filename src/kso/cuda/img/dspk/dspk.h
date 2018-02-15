/*
 * dspk_cuda.h
 *
 *  Created on: Jan 22, 2018
 *      Author: byrdie
 */

#ifndef DSPK_H_
#define DSPK_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cuda.h>

#include "med.h"
#include "norm.h"
#include "gdev.h"
#include "nsd.h"
#include "gm.h"
#include "lm.h"
#include "dspk_util.h"
#include "util/util.h"
#include "util/stride.h"

#include "pyboost.h"

namespace kso {

namespace img {

namespace dspk {

void denoise(buf * data_buf, float med_dev, float std_dev, uint Niter);
void denoise_ndarr(const np::ndarray & data, const np::ndarray & goodmap, float med_dev, float std_dev, uint k_sz, uint Niter);

np::ndarray denoise_fits_file(py::str path, float med_dev, float std_dev, uint k_sz, uint Niter);
np::ndarray denoise_fits_file_quartiles(const np::ndarray & q1, const np::ndarray & q2, const np::ndarray & q3, uint k_sz);









}

}

}


#endif /* DSPK_CUDA_H_ */
