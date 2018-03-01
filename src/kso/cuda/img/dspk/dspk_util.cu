
#include "dspk_util.h"

namespace kso {

namespace img {

namespace dspk {


buf::buf(float * data, float * goodmap, dim3 data_sz, uint kern_sz, dim3 hist_sz, uint n_threads){

	dt = data;
	gm = goodmap;

	sz = data_sz;

	prep(kern_sz, hist_sz, n_threads);



}

buf::buf(std::string path, uint max_sz, uint kern_sz, dim3 hist_sz, uint n_threads){

	// allocate host page-locked memory
	CHECK(cudaHostAlloc(&dt, max_sz * sizeof(float), cudaHostRegisterDefault));
	CHECK(cudaHostAlloc(&gm, sz3 * sizeof(float), cudaHostRegisterDefault));
	CHECK(cudaHostAlloc(&newBad, sizeof(uint), cudaHostRegisterDefault));

	sz = instrument::IRIS::read_fits_raster(path, dt);

	prep(kern_sz, hist_sz, n_threads);

}


void buf::prep(uint kern_sz, dim3 hist_sz, uint n_threads){

	ndim = 3;
	nmet = ndim + 1;	// number of metrics to compare

	mem_fill = 0.5;

	sz3 = sz.x * sz.y * sz.z;

	st.x = 1;
	st.y = sz.x * st.x;
	st.z = sz.y * st.y;

	sb.x = st.x * sizeof(float);
	sb.y = st.y * sizeof(float);
	sb.z = st.z * sizeof(float);

	ksz = kern_sz;

	// Determine how much memory is available on this device
	float mem_fill = 0.5;
	uint device = 0;
	uint mem = mem_fill * kso::util::get_device_mem(device);

	// calculate chunking of input data
	//	uint n_threads = 2;		// Number of host threads
	const uint n_buf = 8;		// number of unique buffers. THIS NUMBER IS HARDCODED. MAKE SURE TO CHANGE IF NEEDED!
	uint t_mem = mem / n_threads;	// Amount of memory per thead
	uint c_mem = t_mem / n_buf;		// Amount of memory per chunk per thread
	uint f_mem = sz.y * sz.x * sizeof(float); 	// Amount of memory occupied by a single frame (spectra / space)

	// save chunk sizes
	csz.x = sz.x;
	csz.y = sz.y;
	csz.z = c_mem / f_mem;		// Max number of frames per chunk
	csz3 = csz.z * csz.x * csz.y;		// number of elements in chunk

	// strides for chunk
	cst.x = 1;
	cst.y = csz.x * cst.x;
	cst.z = csz.y * cst.y;

	hsz = hist_sz;
	hst.x = 1;
	hst.y = sz.x * hst.x;
	hst.z = 0;
	hsz3 = hsz.x * hsz.y;

	printf("Total memory allocated: %.0f MiB\n", mem / pow(2,20) );
	printf("Memory per thread: %.0f MiB\n", t_mem / pow(2,20));
	printf("Memory per chunk: %.0f MiB\n", c_mem / pow(2,20));
	printf("Memory per frame: %.3f MiB\n", f_mem / pow(2,20));
	printf("Number of frames per chunk: %d\n", csz.z);


	// allocate memory on device
	CHECK(cudaMalloc((float **) &buf_d, n_buf * csz3 * sizeof(float)));

	float * dev[n_buf];
	for(uint i = 0; i < n_buf; i++) dev[i] = buf_d + i * csz3;	// initialize array of device buffers

	dt_d = dev[0];
	gm_d = dev[1];

	q2_d = dev[2];	// this array takes nmet size, so the next (four) slots

	gdev_d = dev[5];
	norm_d = dev[6];
	tmp_d = dev[7];



//	CHECK(cudaMalloc((float **) &dt_d, csz3 * sizeof(float)));
//	CHECK(cudaMalloc((float **) &gm_d, csz3 * sizeof(float)));
//	CHECK(cudaMalloc((float **) &q2_d, ndim * csz3 * sizeof(float)));
//	CHECK(cudaMalloc((float **) &gdev_d, ndim * csz3 * sizeof(float)));
//	CHECK(cudaMalloc((float **) &nsd_d, csz3 * sizeof(float)));
//	CHECK(cudaMalloc((float **) &tmp_d, csz3 * sizeof(float)));
//	CHECK(cudaMalloc((float **) &norm_d, ndim * csz3 * sizeof(float)));
//	CHECK(cudaMalloc((float **) &norm_d, csz3 * sizeof(float)));
	CHECK(cudaMalloc((uint **) &newBad_d, sizeof(uint)));
	CHECK(cudaMalloc((float **) &ht_d, nmet * hsz3 * sizeof(float)));
	CHECK(cudaMalloc((float **) &cs_d, nmet * hsz3 * sizeof(float)));
	CHECK(cudaMalloc((float **) &t0_d, nmet * hsz.x * sizeof(float)));
	CHECK(cudaMalloc((float **) &t1_d, nmet * hsz.x * sizeof(float)));
	CHECK(cudaMalloc((float **) &T0_d, nmet * hsz.x * sizeof(float)));
	CHECK(cudaMalloc((float **) &T1_d, nmet * hsz.x * sizeof(float)));

	// calculate offset for kernel
	ks2 = ksz / 2;

	// calculate striding
	S = new kso::util::stride(sz.z, csz.z, 2 * ks2, cst.z);

}

}

}

}
