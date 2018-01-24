
#include "dspk_cuda.h"

using namespace std;


np::ndarray kso::img::dspk::locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter){


	// shape of input data
	uint sz_t = cube.get_shape()[0];
	uint sz_y = cube.get_shape()[1];
	uint sz_l = cube.get_shape()[2];
	uint sz = sz_t * sz_y * sz_l;
	dim3 sz3(sz_l, sz_y, sz_t);

	// stride of input data
	uint n_t = cube.get_strides()[0] / sizeof(float);
	uint n_y = cube.get_strides()[1] / sizeof(float);
	uint n_l = cube.get_strides()[2] / sizeof(float);


	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// initialize goodmap
	float * gm = new float[sz];
	fill(gm, gm + sz, 1.0);

	// initialize neighborhood deviation
	float * dev = new float[sz];

	// storage for the number of bad pixels found on each iteration
	uint newBad = 0;
	uint totBad = 0;

	// allocate pointers for device data
	float * dt_d, * gm_d, * b0_d, *b1_d;
	uint * newBad_d;

	// allocate memory on device
	CHECK(cudaMalloc((float **) &dt_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &gm_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &b0_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &b1_d, sz * sizeof(float)));
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

		// -----------------------------------------------------------------------
		// NEIGHBORHOOD MEAN CONVOLUTION
		kso::img::dspk::calc_dev_x<<<blocks, threads>>>(dt_d, gm_d, b0_d, sz3, k_sz);
		kso::img::dspk::calc_dev_y<<<blocks, threads>>>(gm_d, b0_d, b1_d, sz3, k_sz);
		kso::img::dspk::calc_dev_z<<<blocks, threads>>>(dt_d, gm_d, b1_d, b0_d, sz3, k_sz);
		CHECK(cudaDeviceSynchronize());


		// -----------------------------------------------------------------------
		// NEIGHBORHOOD STANDARD DEVIATION CONVOLUTION
		kso::img::dspk::calc_goodmap<<<blocks, threads>>>(std_dev, gm_d, b0_d, sz3, k_sz, newBad_d);
		CHECK(cudaDeviceSynchronize());


		CHECK(cudaMemcpy(&newBad, newBad_d, sizeof(uint), cudaMemcpyDeviceToHost));
		cout << "Iteration " << iter << ": found " << newBad << " bad pixels\n";
		totBad = totBad + newBad;

	}

	// copy back from devicecudaMemcpyDeviceToHost
	CHECK(cudaMemcpy(gm, gm_d, sz * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(dev, b0_d, sz * sizeof(float), cudaMemcpyDeviceToHost));

	cout << "Total bad pixels: " << totBad << endl;

	// prepare to return Numpy array
	p::object gm_own = p::object();
	p::tuple gm_stride = p::make_tuple(n_t * sizeof(float), n_y * sizeof(float), n_l * sizeof(float));
	p::tuple gm_shape = p::make_tuple(sz_t, sz_y, sz_l);
	np::dtype gm_type = np::dtype::get_builtin<float>();
	np::ndarray gm_arr = np::from_data(gm, gm_type, gm_shape, gm_stride, gm_own);


	return gm_arr;


}

__global__ void kso::img::dspk::calc_dev_x(float * dt, float * gm, float * out, dim3 sz, uint k_sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// retrieve coordinates from thread and block id.
	uint l = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint t = blockIdx.z * blockDim.z + threadIdx.z;

	// retrieve sizes
	uint sz_l = sz.x;
	uint sz_y = sz.y;

	// compute stride sizes
	uint n_l = 1;
	uint n_y = n_l * sz_l;
	uint n_t = n_y * sz_y;


	// initialize neighborhood mean
	float mean = 0.0;
	float norm = 0.0;


	// convolve over spectrum
	for(uint c = 0; c < k_sz; c++){

		// calculate offset
		uint C = l - ks2 + c;

		// truncate kernel if we're over the edge
		if(C >= (sz_l - 1)){
			continue;
		}

		// load from memory
		double gm_0 = gm[n_t * t + n_y * y + n_l * C];
		double dt_0 = dt[n_t * t + n_y * y + n_l * C];

		// update value of mean
		norm = norm + gm_0;
		mean = mean + (gm_0 * dt_0);

	}


	out[n_t * t + n_y * y + n_l * l] =  mean / norm;


}

__global__ void kso::img::dspk::calc_dev_y(float * gm, float * in, float * out, dim3 sz, uint k_sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// retrieve coordinates from thread and block id.
	uint l = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint t = blockIdx.z * blockDim.z + threadIdx.z;

	// retrieve sizes
	uint sz_l = sz.x;
	uint sz_y = sz.y;

	// compute stride sizes
	uint n_l = 1;
	uint n_y = n_l * sz_l;
	uint n_t = n_y * sz_y;


	// initialize neighborhood mean
	float mean = 0.0;
	float norm = 0.0;



	// convolve over space
	for(uint b = 0; b < k_sz; b++){

		// calculate offset
		uint B = y - ks2 + b;

		// truncate kernel if we're over the edge
		if(B >= (sz_y - 1)) {
			continue;
		}


		// load from memory
		double gm_0 = gm[n_t * t + n_y * B + n_l * l];
		double dt_0 = in[n_t * t + n_y * B + n_l * l];

		// update value of mean
		norm = norm + gm_0;
		mean = mean + (gm_0 * dt_0);

	}


	out[n_t * t + n_y * y + n_l * l] =  mean / norm;


}

__global__ void kso::img::dspk::calc_dev_z(float * dt, float * gm, float * in, float * dev, dim3 sz, uint k_sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// retrieve coordinates from thread and block id.
	uint l = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint t = blockIdx.z * blockDim.z + threadIdx.z;

	// retrieve sizes
	uint sz_l = sz.x;
	uint sz_y = sz.y;
	uint sz_t = sz.z;

	// compute stride sizes
	uint n_l = 1;
	uint n_y = n_l * sz_l;
	uint n_t = n_y * sz_y;




	// initialize neighborhood mean
	float mean = 0.0;
	float norm = 0.0;


	// convolve over time
	for(uint a = 0; a < k_sz; a++){

		// calculate offsets
		uint A = t - ks2 + a;

		// truncate the kernel if we're over the edge
		if(A >= (sz_t - 1)){
			continue;
		}


		// load from memory
		double gm_0 = gm[n_t * A + n_y * y + n_l * l];
		double dt_0 = in[n_t * A + n_y * y + n_l * l];

		// update value of mean
		norm = norm + gm_0;
		mean = mean + (gm_0 * dt_0);



	}

	dev[n_t * t + n_y * y + n_l * l] =  dt[n_t * t + n_y * y + n_l * l] - (mean / norm);


}

__global__ void kso::img::dspk::calc_goodmap(float std_dev, float * gm, float * dev, dim3 sz, uint k_sz, uint * new_bad){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// retrieve coordinates from thread and block id.
	uint l = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint t = blockIdx.z * blockDim.z + threadIdx.z;

	// retrieve sizes
	uint sz_l = sz.x;
	uint sz_y = sz.y;
	uint sz_t = sz.z;

	// compute stride sizes
	uint n_l = 1;
	uint n_y = n_l * sz_l;
	uint n_t = n_y * sz_y;


	// initialize neighborhood mean
	float std = 0.0;
	float norm = 0.0;


	// convolve over time
	for(uint a = 0; a < k_sz; a++){

		// calculate offsets
		uint A = t - ks2 + a;

		// truncate the kernel if we're over the edge
		if(A >= (sz_t - 1)){
			continue;
		}



		// convolve over space
		for(uint b = 0; b < k_sz; b++){

			// calculate offset
			uint B = y - ks2 + b;

			// truncate kernel if we're over the edge
			if(B >= (sz_y - 1)) {
				continue;
			}

			// convolve over spectrum
			for(uint c = 0; c < k_sz; c++){

				// calculate offset
				uint C = l - ks2 + c;

				// truncate kernel if we're over the edge
				if(C >= (sz_l - 1)){
					continue;
				}

				// load from memory
				double gm_0 = gm[n_t * A + n_y * B + n_l * C];
				double dev_0 = dev[n_t * A + n_y * B + n_l * C];

				//								cout << dev_0 << endl;

				// update value of mean
				norm = norm + gm_0;
				std = std + (gm_0 * dev_0 * dev_0);

			}

		}

	}

	// finish calculating neighborhood standard deviation
	std = sqrt(std / norm);

	// load from memory
	float dev_0 = dev[n_t * t + n_y * y + n_l * l];
	float gm_0 = gm[n_t * t + n_y * y + n_l * l];

	__syncthreads();

	// check if bad pixel
	if((dev_0 * gm_0) > (std_dev * std)){
		gm[n_t * t + n_y * y + n_l * l] = 0.0;	// update good pixel map
		atomicAdd(new_bad, 1);	// update bad pixel count for this iteration
	}

}

BOOST_PYTHON_MODULE(dspk_cuda){

	//	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	boost::python::def("locate_noise_3D", kso::img::dspk::locate_noise_3D);

}


