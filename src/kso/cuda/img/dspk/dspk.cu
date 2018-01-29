
#include "dspk.h"

using namespace std;

namespace kso {

namespace img {

namespace dspk {

np::ndarray locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter){


	kso::util::enum_device();

	// shape of input data
	uint sz_t = cube.get_shape()[0];
	uint sz_y = cube.get_shape()[1];
	uint sz_l = cube.get_shape()[2];
	uint sz = sz_t * sz_y * sz_l;

	// GPU information
	uint device = 0;
	float mem_fill = 0.5;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	size_t tot_mem = deviceProp.totalGlobalMem;
	size_t mem = tot_mem * mem_fill;

	// calculate chunking of input data
	uint n_threads = 1;		// Number of host threads
	uint n_buf = 6;		// number of unique buffers. THIS NUMBER IS HARDCODED. MAKE SURE TO CHANGE IF NEEDED!
	uint t_mem = mem / n_threads;	// Amount of memory per thead
	uint c_mem = t_mem / n_buf;		// Amount of memory per chunk per thread
	uint f_mem = sz_y * sz_l * sizeof(float); 	// Amount of memory occupied by a single frame (spectra / space)
	uint csz_t = c_mem / f_mem;		// Max number of frames per chunk
	uint N_t = ceil((float) (sz_t) / (float) (csz_t));	// Number of chunks per observation


	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// initialize goodmap
	float * gm = new float[sz];
	float * gdev = new float[sz];
	float * nsd = new float[sz];
	fill(gm, gm + sz, 1.0);



	// storage for the number of bad pixels found on each iteration
	uint newBad = 0;
	uint totBad = 0;

	// allocate pointers for device data
	float * dt_d, * gm_d, * gdev_d, *nsd_d, *tmp_d, *norm_d;
	uint * newBad_d;

	// allocate memory on device
	CHECK(cudaMalloc((float **) &dt_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &gm_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &gdev_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &nsd_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &tmp_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &norm_d, sz * sizeof(float)));
	CHECK(cudaMalloc((uint **) &newBad_d, sizeof(uint)));

	// copy memory to device
	CHECK(cudaMemcpy(dt_d, dt, sz * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(gm_d, gm, sz * sizeof(float), cudaMemcpyHostToDevice));;

	// number of blocks and threads
	dim3 threads(sz_l, 1, 1);
	dim3 blocks(1, sz_y, sz_t);


	// Number of identification iterations
	for(uint iter = 0; iter < Niter; iter++){

		newBad = 0;	// reset the number of bad pixels found for this iteration
		CHECK(cudaMemcpy(newBad_d, &newBad, sizeof(uint), cudaMemcpyHostToDevice));

		kso::img::dspk::calc_norm_0<<<blocks, threads>>>(norm_d, gm_d, sz3, k_sz);
		kso::img::dspk::calc_norm_1<<<blocks, threads>>>(tmp_d, norm_d, sz3, k_sz);
		kso::img::dspk::calc_norm_2<<<blocks, threads>>>(norm_d, tmp_d, sz3, k_sz);

		kso::img::dspk::calc_gdev_0<<<blocks, threads>>>(gdev_d, dt_d, gm_d, sz3, k_sz);
		kso::img::dspk::calc_gdev_1<<<blocks, threads>>>(tmp_d, gdev_d, sz3, k_sz);
		kso::img::dspk::calc_gdev_2<<<blocks, threads>>>(gdev_d, tmp_d, dt_d, gm_d, norm_d, sz3, k_sz);

		kso::img::dspk::calc_nsd_0<<<blocks, threads>>>(nsd_d, gdev_d, sz3, k_sz);
		kso::img::dspk::calc_nsd_1<<<blocks, threads>>>(tmp_d, nsd_d, sz3, k_sz);
		kso::img::dspk::calc_nsd_2<<<blocks, threads>>>(nsd_d, tmp_d, norm_d, sz3, k_sz);

		kso::img::dspk::calc_gm<<<blocks, threads>>>(gm_d, gdev_d, nsd_d, std_dev, newBad_d, sz3, k_sz);



		CHECK(cudaDeviceSynchronize());


		CHECK(cudaMemcpy(&newBad, newBad_d, sizeof(uint), cudaMemcpyDeviceToHost));
		cout << "Iteration " << iter << ": found " << newBad << " bad pixels\n";
		totBad = totBad + newBad;

	}

	// copy back from devicecudaMemcpyDeviceToHost
	CHECK(cudaMemcpy(gm, gm_d, sz * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(gdev, gdev_d, sz * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(nsd, nsd_d, sz * sizeof(float), cudaMemcpyDeviceToHost));

	cout << "Total bad pixels: " << totBad << endl;

	// stride of input data
	uint n_t = cube.get_strides()[0];
	uint n_y = cube.get_strides()[1];
	uint n_l = cube.get_strides()[2];

	// prepare to return Numpy array
	p::object gm_own = p::object();
	p::tuple gm_stride = p::make_tuple(n_t, n_y, n_l);
	p::tuple gm_shape = p::make_tuple(sz_t, sz_y, sz_l);
	np::dtype gm_type = np::dtype::get_builtin<float>();
	np::ndarray gm_arr = np::from_data(gm, gm_type, gm_shape, gm_stride, gm_own);


	return gm_arr;


}









}

}

}





BOOST_PYTHON_MODULE(libkso_cuda){

	//	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	boost::python::def("locate_noise_3D", kso::img::dspk::locate_noise_3D);

}


