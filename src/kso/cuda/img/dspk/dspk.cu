
#include "dspk.h"

using namespace std;

namespace kso {

namespace img {

namespace dspk {

void denoise(buf * data_buf, float std_dev, uint Niter){

	buf * db = data_buf;


	uint ksz = db->ksz;

	float * dt = db->dt;
	float * gm = db->gm;
	uint * newBad = db->newBad;

	float * dt_d = db->dt_d;
	float * gm_d = db->gm_d;
	float * gdev_d = db->gdev_d;
	float * nsd_d = db->nsd_d;
	float * tmp_d = db->tmp_d;
	float * norm_d = db->norm_d;
	uint * newBad_d = db->newBad_d;

	uint num_strides = db->S->num_strides;
	uint * A = db->S->A;
	uint * a = db->S->a;
	uint * L = db->S->L;
	uint * l = db->S->l;
	uint * B = db->S->B;
	uint * b = db->S->b;
	uint * M = db->S->M;
	uint * m = db->S->m;
	uint * b_d = db->S->b_d;

	dim3 sz;
	dim3 threads, blocks;

	uint totBad = 0;

	cout << num_strides << endl;


	// loop over chunks
	for(uint s = 0; s < num_strides; s++){

		cout << A[s] << endl;
		cout << L[s] << endl;
		cout << a[s] << endl;
		cout << l[s] << endl;

		cout << "-----------------" << endl;

		// calculate size for this iteration
		sz.x = db->csz.x;
		sz.y = db->csz.y;
		sz.z = db->S->L[s];

		// calculate number of threads in each dimension
		threads.x = sz.x;
		threads.y = 1;
		threads.z = 1;

		// calculate number of blocks in each dimension
		blocks.x = 1;
		blocks.y = sz.y;
		blocks.z = sz.z;


		// copy memory to device
		CHECK(cudaMemcpy(dt_d, dt + B[s], M[s] * sizeof(float), cudaMemcpyHostToDevice));
//		CHECK(cudaMemcpy(gm_d, gm + B[s], M[s] * sizeof(float), cudaMemcpyHostToDevice));


		// initialize good pixel map
		kso::img::dspk::init_gm<<<blocks, threads>>>(gm_d, sz);

		// Number of identification iterations
		for(uint iter = 0; iter < Niter; iter++){


			*newBad = 0;	// reset the number of bad pixels found for this iteration
			CHECK(cudaMemcpy(newBad_d, newBad, sizeof(uint), cudaMemcpyHostToDevice));

			kso::img::dspk::calc_norm_0<<<blocks, threads>>>(norm_d, gm_d, newBad_d, sz, ksz);
			kso::img::dspk::calc_norm_1<<<blocks, threads>>>(tmp_d, norm_d, sz, ksz);
			kso::img::dspk::calc_norm_2<<<blocks, threads>>>(norm_d, tmp_d, sz, ksz);

			kso::img::dspk::calc_gdev_0<<<blocks, threads>>>(gdev_d, dt_d, gm_d, sz, ksz);
			kso::img::dspk::calc_gdev_1<<<blocks, threads>>>(tmp_d, gdev_d, sz, ksz);
			kso::img::dspk::calc_gdev_2<<<blocks, threads>>>(gdev_d, tmp_d, dt_d, gm_d, norm_d, sz, ksz);

			kso::img::dspk::calc_nsd_0<<<blocks, threads>>>(nsd_d, gdev_d, sz, ksz);
			kso::img::dspk::calc_nsd_1<<<blocks, threads>>>(tmp_d, nsd_d, sz, ksz);
			kso::img::dspk::calc_nsd_2<<<blocks, threads>>>(nsd_d, tmp_d, norm_d, sz, ksz);

			kso::img::dspk::calc_gm<<<blocks, threads>>>(gm_d, gdev_d, nsd_d, std_dev, newBad_d, sz, ksz);



			CHECK(cudaDeviceSynchronize());


			CHECK(cudaMemcpy(newBad, newBad_d, sizeof(uint), cudaMemcpyDeviceToHost));
			cout << "Iteration " << iter << ": found " << *newBad << " bad pixels\n";
			totBad = totBad + *newBad;

		}


		// copy back from devicecudaMemcpyDeviceToHost;
		CHECK(cudaMemcpy(gm + b[s], gm_d + b_d[s], m[s] * sizeof(float), cudaMemcpyDeviceToHost));

		cout << "Total bad pixels: " << totBad << endl;

	}



	return;


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
	float * dn = (float *) denoised_data.get_data();

	uint n_threads = 1;

	buf * db = new buf(dt, dn, sz, k_sz, n_threads);

	denoise(db, std_dev, Niter);

}




}

}

}







