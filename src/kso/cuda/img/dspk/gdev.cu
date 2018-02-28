
#include "gdev.h"


namespace kso {

namespace img {

namespace dspk {

__device__ float gdev_kern_1D(float X, float ks2, float sig){

		float x = X - ks2;

		float var = sig * sig;

		float x2 = x * x;

		return exp(-x2 / var);

}

__global__ void calc_gdev_0(float * gdev_0, float * dt, float * gm, dim3 sz, uint k_sz){

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
	float mean = 0.0;


	// convolve over spectrum
	for(uint c = 0; c < k_sz; c++){

		// calculate offset
		uint C = l - ks2 + c;

		// truncate kernel if we're over the edge
		if(C > (sz_l - 1)){
			continue;
		}

		// calculate kernel at this point
		float k_i = kfac_l * gdev_kern_1D(c, ks2, ksig_l);
//				printf("%d %f\n", c, k_i);

		// load from memory
		float gm_i = gm[n_t * t + n_y * y + n_l * C];
		float dt_i = dt[n_t * t + n_y * y + n_l * C];

		// update value of mean
		mean = mean + (k_i * gm_i * dt_i);

	}


	gdev_0[n_t * t + n_y * y + n_l * l] = mean;
}
__global__ void calc_gdev_1(float * gdev_1, float * gdev_0, dim3 sz, uint k_sz){

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
	float mean = 0.0;

	// convolve over space
	for(uint b = 0; b < k_sz; b++){

		// calculate offset
		uint B = y - ks2 + b;

		// truncate kernel if we're over the edge
		if(B > (sz_y - 1)) {
			continue;
		}

		// calculate kernel at this point
		float k_i = kfac_y * gdev_kern_1D(b, ks2, ksig_y);
//		printf("%d %f\n", b, k_i);


		// load from memory
		float gdev_i = gdev_0[n_t * t + n_y * B + n_l * l];

		// update value of mean
		mean = mean + (k_i * gdev_i);

	}


	gdev_1[n_t * t + n_y * y + n_l * l] =  mean;

}
__global__ void calc_gdev_2(float * gdev_2, float * gdev_1, float * dt, float * gm, float * norm, dim3 sz, uint k_sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;


	// retrieve sizes
	uint sz_l = sz.x;
	uint sz_y = sz.y;
	uint sz_t = sz.z;

	// compute stride sizes
	uint n_l = 1;
	uint n_y = n_l * sz_l;
	uint n_t = n_y * sz_y;

	// retrieve coordinates from thread and block id.
	uint l = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint t = blockIdx.z * blockDim.z + threadIdx.z;


	// initialize neighborhood mean
	float mean = 0.0;

	// convolve over time
	for(uint a = 0; a < k_sz; a++){

		// calculate offsets
		uint A = t - ks2 + a;

		// truncate the kernel if we're over the edge
		if(A > (sz_t - 1)){
			continue;
		}

		// calculate kernel at this point
		float k_i = kfac_t * gdev_kern_1D(a, ks2, ksig_t);
//		printf("%f\n", k_i);


		// load from memory
		float gdev_i = gdev_1[n_t * A + n_y * y + n_l * l];

		// update value of mean
		mean = mean + (k_i * gdev_i);



	}

	float dt_i = dt[n_t * t + n_y * y + n_l * l];
	float gm_i = gm[n_t * t + n_y * y + n_l * l];
	float norm_i = norm[n_t * t + n_y * y + n_l * l];

	gdev_2[n_t * t + n_y * y + n_l * l] = mean / norm_i;
//	if(norm_i != 0.0f){
//		gdev_2[n_t * t + n_y * y + n_l * l] = mean / norm_i;
//	} else {
//		gdev_2[n_t * t + n_y * y + n_l * l] = 0.0f;
//	}

}



}

}

}
