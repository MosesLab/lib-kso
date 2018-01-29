
#include "nsd.h"



namespace kso {

namespace img {

namespace dspk {

__global__ void calc_nsd_0(float * nsd_0, float * gdev, dim3 sz, uint k_sz){
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
	float nsd = 0.0;

	// convolve over spectrum
	for(uint c = 0; c < k_sz; c++){

		// calculate offset
		uint C = l - ks2 + c;

		// truncate kernel if we're over the edge
		if(C > (sz_l - 1)){
			continue;
		}

		// load from memory
		double dev_i = gdev[n_t * t + n_y * y + n_l * C];

		//								cout << dev_0 << endl;

		// update value of mean
		nsd = nsd + (dev_i * dev_i);

	}


	// finish calculating neighborhood standard deviation
	nsd_0[n_t * t + n_y * y + n_l * l] = nsd;


}
__global__ void calc_nsd_1(float * nsd_1, float * nsd_0, dim3 sz, uint k_sz){
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
	float nsd = 0.0;

	// convolve over space
	for(uint b = 0; b < k_sz; b++){

		// calculate offset
		uint B = y - ks2 + b;

		// truncate kernel if we're over the edge
		if(B > (sz_y - 1)) {
			continue;
		}


		// load from memory
		double nsd_i = nsd_0[n_t * t + n_y * B + n_l * l];

		//								cout << dev_0 << endl;

		// update value of mean
		nsd = nsd + nsd_i;

	}


	// finish calculating neighborhood standard deviation
	nsd_1[n_t * t + n_y * y + n_l * l] = nsd;

}
__global__ void calc_nsd_2(float * nsd_2, float * nsd_1, float * norm, dim3 sz, uint k_sz){
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
	float nsd = 0.0;


	// convolve over time
	for(uint a = 0; a < k_sz; a++){

		// calculate offsets
		uint A = t - ks2 + a;

		// truncate the kernel if we're over the edge
		if(A > (sz_t - 1)){
			continue;
		}

		// load from memory
		double nsd_i = nsd_1[n_t * A + n_y * y + n_l * l];

		//								cout << dev_0 << endl;

		// update value of mean
		nsd = nsd + nsd_i;



	}

	// finish calculating neighborhood standard deviation
	float norm_i = norm[n_t * t + n_y * y + n_l * l];

	nsd_2[n_t * t + n_y * y + n_l * l] = sqrt(nsd / norm_i);

}


}

}

}
