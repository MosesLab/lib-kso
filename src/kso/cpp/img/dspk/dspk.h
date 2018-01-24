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

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

class dim3{
public:
	uint x;
	uint y;
	uint z;
	dim3(uint _x, uint _y, uint _z){
		x = _x;
		y = _y;
		z = _z;

	}
};

//test
namespace p = boost::python;
namespace np = boost::python::numpy;

namespace kso {

namespace img {

namespace dspk {

np::ndarray locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter);

void calc_nm_x(float * dt, float * nm_out,  float * gm, float * nrm_out, dim3 sz, uint k_sz);
void calc_nm_y(float * nm_in, float * nm_out, float * nrm_in, float * nrm_out, dim3 sz, uint k_sz);
void calc_nm_z(float * nm_in, float * nm_out, float * nrm_in, float * nrm_out, dim3 sz, uint k_sz);

void calc_dev(float * dt, float * nm, float * dev, dim3 sz);

void calc_nsd_x(float * dev, float * nsd_out, float * gm, dim3 sz, uint k_sz);
void calc_nsd_y(float * nsd_in, float * nsd_out, dim3 sz, uint k_sz);
void calc_nsd_z(float * nsd_in, float * nsd_out, float * nrm, dim3 sz, uint k_sz);

void update_gm(float std_dev, float * gm, float * dev, float * nsd, dim3 sz, uint * new_bad);

}

}

}


#endif /* DSPK_CUDA_H_ */
