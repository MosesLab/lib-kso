
#include "dspk.h"

using namespace std;

namespace kso {

namespace img {

namespace dspk {

np::ndarray locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter, uint n_threads){


	kso::util::enum_device();

	// shape of input data
	uint dsz_t = cube.get_shape()[0];
	uint dsz_y = cube.get_shape()[1];
	uint dsz_l = cube.get_shape()[2];
	uint dsz = dsz_t * dsz_y * dsz_l;

	// stride of input data
	uint n_t = cube.get_strides()[0] / sizeof(float);
	uint n_y = cube.get_strides()[1] / sizeof(float);
	uint n_l = cube.get_strides()[2] / sizeof(float);

	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// initialize host memory
	float * gm = new float[dsz];
	float * gdev = new float[dsz];
	float * nsd = new float[dsz];
	uint newBad = 0;			// Number of bad pixels found on each iteration
	uint totBad = 0;

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// GPU information
	uint device = 0;
	float mem_fill = 0.5;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	size_t tot_mem = deviceProp.totalGlobalMem;
	size_t mem = tot_mem * mem_fill;

	// calculate chunking of input data
//	uint n_threads = 2;		// Number of host threads
	uint n_buf = 6;		// number of unique buffers. THIS NUMBER IS HARDCODED. MAKE SURE TO CHANGE IF NEEDED!
	uint t_mem = mem / n_threads;	// Amount of memory per thead
	uint c_mem = t_mem / n_buf;		// Amount of memory per chunk per thread
	uint f_mem = dsz_y * dsz_l * sizeof(float); 	// Amount of memory occupied by a single frame (spectra / space)
	uint csz_t = c_mem / f_mem;		// Max number of frames per chunk
	uint N_t = ceil((float) (dsz_t) / (float) (csz_t));	// Number of chunks per observation
	uint csz = csz_t * dsz_y * dsz_l;		// number of elements in chunk

	printf("Total memory allocated: %.0f MiB\n", mem / pow(2,20) );
	printf("Memory per thread: %.0f MiB\n", t_mem / pow(2,20));
	printf("Memory per chunk: %.0f MiB\n", c_mem / pow(2,20));
	printf("Memory per frame: %.3f MiB\n", f_mem / pow(2,20));
	printf("Number of frames per chunk: %d\n", csz_t);
	printf("Number of chunks per observation: %d\n", N_t);


	// number of blocks and threads
	dim3 threads(dsz_l, 1, 1);
	dim3 blocks(1, dsz_y, csz_t);


	// allocate pointers for device data
	float * dt_d, * gm_d, * gdev_d, *nsd_d, *tmp_d, *norm_d;
	uint * newBad_d;

	// allocate memory on device
	CHECK(cudaMalloc((float **) &dt_d, csz * sizeof(float)));
	CHECK(cudaMalloc((float **) &gm_d, csz * sizeof(float)));
	CHECK(cudaMalloc((float **) &gdev_d, csz * sizeof(float)));
	CHECK(cudaMalloc((float **) &nsd_d, csz * sizeof(float)));
	CHECK(cudaMalloc((float **) &tmp_d, csz * sizeof(float)));
	CHECK(cudaMalloc((float **) &norm_d, csz * sizeof(float)));
	CHECK(cudaMalloc((uint **) &newBad_d, sizeof(uint)));

	// calculate chunking
	uint num_strides = kso::util::stride::get_num_strides(dsz_t, csz_t, 2 * ks2);
	uint * A = new uint[num_strides];
	uint * a = new uint[num_strides];
	uint * L = new uint[num_strides];
	uint * l = new uint[num_strides];
	uint * a_d = new uint[num_strides];
	kso::util::stride::get_strides(dsz_t, csz_t, 2 * ks2, A, a, L, l, a_d);

	cout << "num_strides: " << num_strides << endl;



	for(uint S = 0; S < num_strides; S++){

		cout << A[S] << endl;
		cout << L[S] << endl;
		cout << a[S] << endl;
		cout << l[S] << endl;

		cout << "-----------------" << endl;


		// final size of chunk as seen by kernel
		uint sz_l = dsz_l;
		uint sz_y = dsz_y;
		uint sz_t = L[S];


		// calculate final memory offset
		uint B = A[S] * n_t;

		// copy sizes into dim3 object
		dim3 sz3(sz_l, sz_y, sz_t);
		uint sz = sz_l * sz_y * sz_t;



		// copy memory to device
		CHECK(cudaMemcpy(dt_d, dt + B, sz * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(gm_d, gm + B, sz * sizeof(float), cudaMemcpyHostToDevice));


		// initialize good pixel map
		kso::img::dspk::init_gm<<<blocks, threads>>>(gm_d, sz3);

		// Number of identification iterations
		for(uint iter = 0; iter < Niter; iter++){

			newBad = 0;	// reset the number of bad pixels found for this iteration
//			CHECK(cudaMemcpy(newBad_d, &newBad, sizeof(uint), cudaMemcpyHostToDevice));

			kso::img::dspk::calc_norm_0<<<blocks, threads>>>(norm_d, gm_d, newBad_d, sz3, k_sz);
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


//			CHECK(cudaMemcpy(&newBad, newBad_d, sizeof(uint), cudaMemcpyDeviceToHost));
			cout << "Iteration " << iter << ": found " << newBad << " bad pixels\n";
			totBad = totBad + newBad;

		}

		uint b = a[S] * n_t;
		uint gsz = l[S] * n_t;
		uint b_d = a_d[S] * n_t;

		// copy back from devicecudaMemcpyDeviceToHost;
		CHECK(cudaMemcpy(gm + b, gm_d + b_d, gsz * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(gdev + b, gdev_d + b_d, gsz * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(nsd + b, nsd_d + b_d, gsz * sizeof(float), cudaMemcpyDeviceToHost));

		cout << "Total bad pixels: " << totBad << endl;

	}



	// prepare to return Numpy array
	p::object gm_own = p::object();
	p::tuple gm_stride = p::make_tuple(n_t * sizeof(float), n_y * sizeof(float), n_l * sizeof(float));
	p::tuple gm_shape = p::make_tuple(dsz_t, dsz_y, dsz_l);
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

