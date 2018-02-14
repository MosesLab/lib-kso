
#include "gm.h"


namespace kso {

namespace img {

namespace dspk {

__global__ void calc_gm(float * q1, float * q2, float * q3,
		float * dt, float * gm, uint * new_bad,
		dim3 sz, dim3 ksz, float fence){

	// calculate offset for kernel
	dim3 kr;
	kr.x = ksz.x / 2;
	kr.y = ksz.y / 2;
	kr.z = ksz.z / 2;

	// compute stride sizes
	dim3 n;
	n.x = 1;
	n.y = n.x * sz.x;
	n.z = n.y * sz.y;

	// retrieve coordinates from thread and block id.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	// overall index
	uint L = n.z * z + n.y * y + n.x * x;

	// calculate data devation from mean and mask with goodmap
	float gdev = (dt[L] - q2[L]) * gm[L];

	// calculate interquartile range
	float iqr_12 = q2[L] - q1[L];
	float iqr_23 = q3[L] - q2[L];
//	float iqr = q3[L] - q1[L];
//	float iqr_23 = iqr;
//	float iqr_12 = iqr;

	// check if bad pixel
	if(gdev > (fence * iqr_23)){
		gm[L] = 0.0f;
		dt[L] = 0.0f;
		atomicAdd(new_bad, 1);
	} else if (gdev < -(fence * iqr_12)) {
		gm[L] = 0.0f;
		dt[L] = 0.0f;
		atomicAdd(new_bad, 1);
	}


}

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

	gm[n_t * t + n_y * y + n_l * l] = 1.0f;	// update good pixel map

//	if(dt[n_t * t + n_y * y + n_l * l] < 0.0){
//		gm[n_t * t + n_y * y + n_l * l] = 0.0;	// update good pixel map
//	} else {
//		gm[n_t * t + n_y * y + n_l * l] = 1.0;	// update good pixel map
//	}



}


}

}

}
