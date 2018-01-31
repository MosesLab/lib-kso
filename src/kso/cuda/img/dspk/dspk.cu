
#include "dspk.h"

using namespace std;

namespace kso {

namespace img {

namespace dspk {

void denoise(buf data_buf, float std_dev, uint Niter){

	buf db = data_buf;

	kso::util::enum_device();


	// shape of input data
	dim3 dsz;
	dsz.z = cube.get_shape()[0];
	dsz.y = cube.get_shape()[1];
	dsz.x = cube.get_shape()[2];
	uint dsz3 = dsz.z * dsz.y * dsz.x;

	// array stride of input data
	dim3 n;


	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// initialize host memory
	float * gm = new float[dsz3];
	float * gdev = new float[dsz3];
	float * nsd = new float[dsz3];
	uint newBad = 0;			// Number of bad pixels found on each iteration
	uint totBad = 0;






	// number of blocks and threads
	dim3 threads(dsz_l, 1, 1);
	dim3 blocks(1, dsz_y, csz_t);






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
	py::object gm_own = py::object();
	py::tuple gm_stride = py::make_tuple(n_t * sizeof(float), n_y * sizeof(float), n_l * sizeof(float));
	py::tuple gm_shape = py::make_tuple(dsz_t, dsz_y, dsz_l);
	np::dtype gm_type = np::dtype::get_builtin<float>();
	np::ndarray gm_arr = np::from_data(gm, gm_type, gm_shape, gm_stride, gm_own);


	return gm_arr;


}

void denoise_ndarr(const np::ndarray & denoised_data, const np::ndarray & data, float std_dev, uint k_sz, uint Niter){

	// shape of input data
	dim3 sz;
	sz.z = data.get_shape()[0];
	sz.y = data.get_shape()[1];
	sz.x = data.get_shape()[2];

	dim3 st;
	st.z = data.get_strides()[0] / sizeof(float);
	st.y = data.get_strides()[1] / sizeof(float);
	st.x = data.get_strides()[2] / sizeof(float);

	// extract float data from numpy array
	float * dt = (float *) data.get_data();

	uint n_threads = 1;

	buf db = new buf(dt, sz, k_sz, n_threads);

	denoise(db, std_dev, Niter);

}




}

}

}







