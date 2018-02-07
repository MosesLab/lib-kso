#include "lm.h"



namespace kso {

namespace img {

namespace dspk {

__device__ float local_kern_1D(uint X, uint ksz){



	// calculate offset for kernel
	uint ks2 = ksz / 2;

	float sig = ks2;
	float var = sig * sig;

	float x = (float) X - (float) ks2;
	float x2 = x * x;

	return exp(-x2 / var) / (1.0 + x2);

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

		// calculate kernel at this point
		float k_i = local_kern_1D(c, k_sz);

		// load from memory
		float gm_i = gm[n_t * t + n_y * y + n_l * C];
		float dt_i = dt[n_t * t + n_y * y + n_l * C];

		// update value of mean
		sum = sum + (gm_i * dt_i * k_i);

	}


	dt_0[n_t * t + n_y * y + n_l * l] = sum;
}


__global__ void calc_dt_1(float * dt_1, float * dt_0, dim3 sz, uint k_sz){

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

	// convolve over space
	for(uint b = 0; b < k_sz; b++){

		// calculate offset
		uint B = y - ks2 + b;

		// truncate kernel if we're over the edge
		if(B > (sz_y - 1)) {
			continue;
		}

		// calculate kernel at this point
		float k_i = local_kern_1D(b, k_sz);

		// load from memory
		float dt_i = dt_0[n_t * t + n_y * B + n_l * l];

		// update value of mean
		sum = sum + (dt_i * k_i);

	}


	dt_1[n_t * t + n_y * y + n_l * l] =  sum;

}
__global__ void calc_dt_2(float * dt_2, float * dt_1, float * gm, float * norm, dim3 sz, uint k_sz){

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
	float sum = 0.0;

	// convolve over time
	for(uint a = 0; a < k_sz; a++){

		// calculate offsets
		uint A = t - ks2 + a;

		// truncate the kernel if we're over the edge
		if(A > (sz_t - 1)){
			continue;
		}

		// calculate kernel at this point
		float k_i = local_kern_1D(a, k_sz);

		// load from memory
		float dt_i = dt_1[n_t * A + n_y * y + n_l * l];

		// update value of mean
		sum = sum + (dt_i * k_i);



	}

	float dt_i = dt_2[n_t * t + n_y * y + n_l * l];
	float gm_i = gm[n_t * t + n_y * y + n_l * l];
	float norm_i = norm[n_t * t + n_y * y + n_l * l];

	float bm_i = 1.0 - gm_i;	// bad pixels are now equal to one.
	float lm_i = sum / norm_i;	// local mean that we've actually been calculating the whole time


	dt_2[n_t * t + n_y * y + n_l * l] = (dt_i * gm_i) + (bm_i * lm_i);
//	dt_2[n_t * t + n_y * y + n_l * l] = (dt_i * gm_i) ;

}

__global__ void calc_lmn_0(float * norm_0, float * gm, uint * bad_pix, dim3 sz, uint k_sz){



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
	float norm = 0.0;


	// convolve over spectrum
	for(uint c = 0; c < k_sz; c++){

		// calculate offset
		uint C = l - ks2 + c;

		// truncate kernel if we're over the edge
		if(C > (sz_l - 1)){
			continue;
		}

		// calculate kernel at this point
		float k_i = local_kern_1D(c, k_sz);

		// load from memory
		float gm_i = gm[n_t * t + n_y * y + n_l * C];

		// update value of mean
		norm = norm + (gm_i * k_i);

	}


	norm_0[n_t * t + n_y * y + n_l * l] = norm;

}
__global__ void calc_lmn_1(float * norm_1, float * norm_0, dim3 sz, uint k_sz){
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
	float norm = 0.0;

	// convolve over space
	for(uint b = 0; b < k_sz; b++){

		// calculate offset
		uint B = y - ks2 + b;

		// truncate kernel if we're over the edge
		if(B > (sz_y - 1)) {
			continue;
		}

		// calculate kernel at this point
		float k_i = local_kern_1D(b, k_sz);

		// load from memory
		float norm_i = norm_0[n_t * t + n_y * B + n_l * l];

		// update value of mean
		norm = norm + (norm_i * k_i);

	}


	norm_1[n_t * t + n_y * y + n_l * l] =  norm;

}
__global__ void calc_lmn_2(float * norm_2, float * norm_1, dim3 sz, uint k_sz){
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
	float norm = 0.0;

	// convolve over time
	for(uint a = 0; a < k_sz; a++){

		// calculate offsets
		uint A = t - ks2 + a;

		// truncate the kernel if we're over the edge
		if(A > (sz_t - 1)){
			continue;
		}

		// calculate kernel at this point
		float k_i = local_kern_1D(a, k_sz);

		// load from memory
		float norm_i = norm_1[n_t * A + n_y * y + n_l * l];

		// update value of mean
		norm = norm + (norm_i * k_i);


	}

	norm_2[n_t * t + n_y * y + n_l * l] = norm;
}


}

}

}


