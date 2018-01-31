
#include "dspk_util.h"

namespace kso {

namespace img {

namespace dspk {


buf::buf(float * data, dim3 data_sz, uint kern_sz, uint n_threads){

	float mem_fill = 0.5;

	dt = data;

	sz = data_sz;
	sz3 = sz.x * sz.y * sz.z;

	ksz = kern_sz;

	// Determine how much memory is available on this device
	uint device = 0;
	uint mem = mem_fill * kso::util::get_device_mem(device);

	// calculate chunking of input data
	//	uint n_threads = 2;		// Number of host threads
	uint n_buf = 6;		// number of unique buffers. THIS NUMBER IS HARDCODED. MAKE SURE TO CHANGE IF NEEDED!
	uint t_mem = mem / n_threads;	// Amount of memory per thead
	uint c_mem = t_mem / n_buf;		// Amount of memory per chunk per thread
	uint f_mem = sz.y * sz.x * sizeof(float); 	// Amount of memory occupied by a single frame (spectra / space)
	csz.x = sz.x;
	csz.y = sz.y;
	csz.t = c_mem / f_mem;		// Max number of frames per chunk
	csz3 = csz.t * csz.x * csz.y;		// number of elements in chunk

	printf("Total memory allocated: %.0f MiB\n", mem / pow(2,20) );
	printf("Memory per thread: %.0f MiB\n", t_mem / pow(2,20));
	printf("Memory per chunk: %.0f MiB\n", c_mem / pow(2,20));
	printf("Memory per frame: %.3f MiB\n", f_mem / pow(2,20));
	printf("Number of frames per chunk: %d\n", csz.t);

	cudaHostRegister(dt, sz3, cudaHostRegisterDefault);

	// allocate memory on device
	CHECK(cudaMalloc((float **) &dt_d, csz3 * sizeof(float)));
	CHECK(cudaMalloc((float **) &gm_d, csz3 * sizeof(float)));
	CHECK(cudaMalloc((float **) &gdev_d, csz3 * sizeof(float)));
	CHECK(cudaMalloc((float **) &nsd_d, csz3 * sizeof(float)));
	CHECK(cudaMalloc((float **) &tmp_d, csz3 * sizeof(float)));
	CHECK(cudaMalloc((float **) &norm_d, csz3 * sizeof(float)));
	CHECK(cudaMalloc((uint **) &newBad_d, sizeof(uint)));

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// calculate striding
	S = new kso::util::stride(dsz_t, csz_t, 2 * ks2, A, a, L, l, a_d);

}




}

}

}
