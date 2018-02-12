
#include "gm.h"


namespace kso {

namespace img {

namespace dspk {

__global__ void calc_gm(float * gm, float * gdev, float * nsd, float std_dev, uint * new_bad, dim3 sz, uint k_sz){
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

	// load from memory
	float gdev_i = gdev[n_t * t + n_y * y + n_l * l];
	float nsd_i = nsd[n_t * t + n_y * y + n_l * l];

	// check if bad pixel
	if((gdev_i) > (std_dev * nsd_i)){
		gm[n_t * t + n_y * y + n_l * l] = 0.0;	// update good pixel map
		atomicAdd(new_bad, 1);
	}
}

__global__ void init_gm(float * gm, float * dt, dim3 sz){

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

	if(dt[n_t * t + n_y * y + n_l * l] < 0.0){
		gm[n_t * t + n_y * y + n_l * l] = 0.0;	// update good pixel map
	} else {
		gm[n_t * t + n_y * y + n_l * l] = 1.0;	// update good pixel map
	}



}


}

}

}
