#include "lm.h"



namespace kso {

namespace img {

namespace dspk {

__device__ float local_kern_1D(uint X, uint ksz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	float x = X - ks2;
	float x2 = x * x;

	return exp(-x2) / (1 + x2);

}


__global__ void calc_dt_0(float * dt_0, float * dt, float * gm, dim3 sz, uint k_sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;


	// retrieve sizes
	uint sz_l = sz.x;
	uint sz_y = sz.y;

	// compute stride sizes
	uint n_l = 1;
	uint n_y = n_l * sz_l;
	uint n_t = n_y * sz_y;

	// retrieve coordinates from thread and block id.
	uint l = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint t = blockIdx.z * blockDim.z + threadIdx.z;


	// initialize neighborhood mean
	float sum = 0.0;


	// convolve over spectrum
	for(uint c = 0; c < k_sz; c++){

		// calculate offset
		uint C = l - ks2 + c;

		// truncate kernel if we're over the edge
		if(C > (sz_l - 1)){
			continue;
		}

		// load from memory
		double gm_i = gm[n_t * t + n_y * y + n_l * C];
		double dt_i = dt[n_t * t + n_y * y + n_l * C];

		// update value of mean
		sum = sum + (gm_i * dt_i);

	}


	gdev_0[n_t * t + n_y * y + n_l * l] = sum;
}

}
__global__ void calc_dt_1(float * dt_1, float * dt_0, dim3 sz, uint k_sz){



}
__global__ void calc_dt_2(float * dt_2, float * dt_1, float * gm, float * norm, dim3 sz, uint k_sz){



}

}

}

}
