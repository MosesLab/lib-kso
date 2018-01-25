
#include "dspk_cuda.h"

using namespace std;


np::ndarray kso::img::dspk::locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter){


	// shape of input data
	uint sz_t = cube.get_shape()[0];
	uint sz_y = cube.get_shape()[1];
	uint sz_l = cube.get_shape()[2];
	uint sz = sz_t * sz_y * sz_l;
	dim3 sz3(sz_l, sz_y, sz_t);




	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// initialize goodmap
	float * gm = new float[sz];
	fill(gm, gm + sz, 1.0);



	// storage for the number of bad pixels found on each iteration
	uint newBad = 0;
	uint totBad = 0;

	// allocate pointers for device data
	float * dt_d, * gm_d, * dev_d, *nsd_d, *buf0_d, *nrm_d, *buf1_d;
	uint * newBad_d;

	// allocate memory on device
	CHECK(cudaMalloc((float **) &dt_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &gm_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &dev_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &nsd_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &buf0_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &nrm_d, sz * sizeof(float)));
	CHECK(cudaMalloc((float **) &buf1_d, sz * sizeof(float)));
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
		kso::img::dspk::calc_nm_x<<<blocks, threads>>>(dt_d, dev_d, gm_d, nrm_d, sz3, k_sz);
		kso::img::dspk::calc_nm_y<<<blocks, threads>>>(dev_d, buf0_d, nrm_d, buf1_d, sz3, k_sz);
		kso::img::dspk::calc_nm_z<<<blocks, threads>>>(buf0_d, dev_d, buf1_d, nrm_d, sz3, k_sz);

		kso::img::dspk::calc_dev<<<blocks, threads>>>(dt_d, dev_d, dev_d, sz3);


		// -----------------------------------------------------------------------
		// NEIGHBORHOOD STANDARD DEVIATION CONVOLUTION
		kso::img::dspk::calc_nsd_x<<<blocks, threads>>>(dev_d, nsd_d, gm_d, sz3, k_sz);
		kso::img::dspk::calc_nsd_y<<<blocks, threads>>>(nsd_d, buf0_d, sz3, k_sz);
		kso::img::dspk::calc_nsd_z<<<blocks, threads>>>(buf0_d, nsd_d, nrm_d, sz3, k_sz);

		kso::img::dspk::update_gm<<<blocks, threads>>>(std_dev, gm_d, dev_d, nsd_d, sz3, newBad_d);

		CHECK(cudaDeviceSynchronize());


		CHECK(cudaMemcpy(&newBad, newBad_d, sizeof(uint), cudaMemcpyDeviceToHost));
		cout << "Iteration " << iter << ": found " << newBad << " bad pixels\n";
		totBad = totBad + newBad;

	}

	// copy back from devicecudaMemcpyDeviceToHost
	CHECK(cudaMemcpy(gm, gm_d, sz * sizeof(float), cudaMemcpyDeviceToHost));

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



__global__ void kso::img::dspk::calc_nm_x(float * dt, float * nm_out,  float * gm, float * nrm_out, dim3 sz, uint k_sz){

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
	float norm = 0.0;


	// convolve over spectrum
	for(uint c = 0; c < k_sz; c++){

		// calculate offset
		uint C = l - ks2 + c;

		// truncate kernel if we're over the edge
		if(C > (sz_l - 1)){
			continue;
		}

		// load from memory
		double gm_0 = gm[n_t * t + n_y * y + n_l * C];
		double dt_0 = dt[n_t * t + n_y * y + n_l * C];

		// update value of mean
		norm = norm + gm_0;
		mean = mean + (gm_0 * dt_0);

	}


	nm_out[n_t * t + n_y * y + n_l * l] = mean;
	nrm_out[n_t * t + n_y * y + n_l * l] = norm;


}

__global__ void kso::img::dspk::calc_nm_y(float * nm_in, float * nm_out, float * nrm_in, float * nrm_out, dim3 sz, uint k_sz){

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
	float norm = 0.0;



	// convolve over space
	for(uint b = 0; b < k_sz; b++){

		// calculate offset
		uint B = y - ks2 + b;

		// truncate kernel if we're over the edge
		if(B > (sz_y - 1)) {
			continue;
		}


		// load from memory
		double nrm_0 = nrm_in[n_t * t + n_y * B + n_l * l];
		double nm_0 = nm_in[n_t * t + n_y * B + n_l * l];

		// update value of mean
		norm = norm + nrm_0;
		mean = mean + nm_0;

	}


	nm_out[n_t * t + n_y * y + n_l * l] =  mean;
	nrm_out[n_t * t + n_y * y + n_l * l] =  norm;




}

__global__ void kso::img::dspk::calc_nm_z(float * nm_in, float * nm_out, float * nrm_in, float * nrm_out, dim3 sz, uint k_sz){

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
	float norm = 0.0;


	// convolve over time
	for(uint a = 0; a < k_sz; a++){

		// calculate offsets
		uint A = t - ks2 + a;

		// truncate the kernel if we're over the edge
		if(A > (sz_t - 1)){
			continue;
		}


		// load from memory
		double nrm_0 = nrm_in[n_t * A + n_y * y + n_l * l];
		double nm_0 = nm_in[n_t * A + n_y * y + n_l * l];

		// update value of mean
		norm = norm + nrm_0;
		mean = mean + nm_0;



	}

	nm_out[n_t * t + n_y * y + n_l * l] = mean / norm;
	nrm_out[n_t * t + n_y * y + n_l * l] = norm;


}

__global__ void kso::img::dspk::calc_dev(float * dt, float * nm, float * dev, dim3 sz){

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

	dev[n_t * t + n_y * y + n_l * l] = dt[n_t * t + n_y * y + n_l * l] - nm[n_t * t + n_y * y + n_l * l];

}



__global__ void kso::img::dspk::calc_nsd_x(float * dev, float * nsd_out, float * gm, dim3 sz, uint k_sz){

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

	// convolve over spectrum
	for(uint c = 0; c < k_sz; c++){

		// calculate offset
		uint C = l - ks2 + c;

		// truncate kernel if we're over the edge
		if(C > (sz_l - 1)){
			continue;
		}

		// load from memory
		double gm_0 = gm[n_t * t + n_y * y + n_l * C];
		double dev_0 = dev[n_t * t + n_y * y + n_l * C];

		//								cout << dev_0 << endl;

		// update value of mean
		nsd = nsd + (gm_0 * dev_0 * dev_0);

	}


	// finish calculating neighborhood standard deviation
	nsd_out[n_t * t + n_y * y + n_l * l] = nsd;




}
__global__ void kso::img::dspk::calc_nsd_y(float * nsd_in, float * nsd_out, dim3 sz, uint k_sz){
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

	// convolve over space
	for(uint b = 0; b < k_sz; b++){

		// calculate offset
		uint B = y - ks2 + b;

		// truncate kernel if we're over the edge
		if(B > (sz_y - 1)) {
			continue;
		}


		// load from memory
		double nsd_0 = nsd_in[n_t * t + n_y * B + n_l * l];

		//								cout << dev_0 << endl;

		// update value of mean
		nsd = nsd + nsd_0;

	}


	// finish calculating neighborhood standard deviation
	nsd_out[n_t * t + n_y * y + n_l * l] = nsd;



}

__global__ void kso::img::dspk::calc_nsd_z(float * nsd_in, float * nsd_out, float * nrm, dim3 sz, uint k_sz){
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
		double nsd_0 = nsd_in[n_t * A + n_y * y + n_l * l];

		//								cout << dev_0 << endl;

		// update value of mean
		nsd = nsd + nsd_0;



	}

	// finish calculating neighborhood standard deviation
	float nrm_0 = nrm[n_t * t + n_y * y + n_l * l];

	nsd_out[n_t * t + n_y * y + n_l * l] = sqrt(nsd / nrm_0);





}

__global__ void kso::img::dspk::update_gm(float std_dev, float * gm, float * dev, float * nsd, dim3 sz, uint * new_bad){
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

	// load from memory
	float dev_0 = dev[n_t * t + n_y * y + n_l * l];
	float nsd_0 = nsd[n_t * t + n_y * y + n_l * l];
	float gm_0 = gm[n_t * t + n_y * y + n_l * l];

	// check if bad pixel
	if((dev_0 * gm_0) > (std_dev * nsd_0)){
		gm[n_t * t + n_y * y + n_l * l] = 0.0;	// update good pixel map
		atomicAdd(new_bad, 1);
	}

}

BOOST_PYTHON_MODULE(dspk_cuda){

	//	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	boost::python::def("locate_noise_3D", kso::img::dspk::locate_noise_3D);

}


