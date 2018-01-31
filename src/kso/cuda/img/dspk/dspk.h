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

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "norm.h"
#include "gdev.h"
#include "nsd.h"
#include "gm.h"
#include "dspk_util.h"
#include "util/util.h"
#include "util/stride.h"

//test
namespace py = boost::python;
namespace np = boost::python::numpy;

namespace kso {

namespace img {

namespace dspk {

void denoise(buf * data_buf, float std_dev, uint Niter);
void denoise_ndarr(const np::ndarray & denoised_data, const np::ndarray & data, float std_dev, uint k_sz, uint Niter);

void * denoise_fits_file(std::string out_path, std::string in_path, float std_dev, uint k_sz, uint Niter);
//void * denoise_fits_file(py::str out_path, py::str in_path, float std_dev, uint k_sz, uint Niter);

void denoise_fits(py::list out_paths, py::list in_paths, float std_dev, uint k_sz, uint Niter);




np::ndarray locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter, uint n_threads);



}

}

}


#endif /* DSPK_CUDA_H_ */
