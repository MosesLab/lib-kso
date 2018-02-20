
#include "dspk.h"

using namespace std;

namespace kso {

namespace img {

namespace dspk {

void denoise(buf * data_buf, float tmax, float tmin, uint Niter){

	buf * db = data_buf;


	uint ksz1 = db->ksz;
	dim3 ksz(ksz1, ksz1, ksz1);

	float * dt = db->dt;
	float * q1 = db->q1;
	float * q2 = db->q2;
	float * q3 = db->q3;
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

	float * q1_d = gdev_d;
	float * q2_d = nsd_d;
	float * q3_d = norm_d;


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
		kso::img::dspk::init_gm<<<blocks, threads>>>(gm_d, dt_d, sz);

		cout << "Median Filter" << endl;

		// Number of identification iterations
		for(uint iter = 0; iter < Niter; iter++){


			*newBad = 0;	// reset the number of bad pixels found for this iteration
			CHECK(cudaMemcpy(newBad_d, newBad, sizeof(uint), cudaMemcpyHostToDevice));

			calc_quartiles(q1_d, q2_d, q3_d, dt_d, gm_d, tmp_d, sz, ksz);
//			calc_gm<<<blocks, threads>>>(q1_d, q2_d, q3_d, dt_d, gm_d, newBad_d, sz, ksz, med_dev);


			CHECK(cudaMemcpy(newBad, newBad_d, sizeof(uint), cudaMemcpyDeviceToHost));
			cout << "Iteration " << iter << ": found " << *newBad << " bad pixels\n";
			totBad = totBad + *newBad;

			if(*newBad == 0){	// stop if we're not finding any pixels
				break;
			}

		}

//		cout << "Mean Filter" << endl;
//
//		// Number of identification iterations
//		for(uint iter = 0; iter < Niter; iter++){
//
//			*newBad = 0;	// reset the number of bad pixels found for this iteration
//			CHECK(cudaMemcpy(newBad_d, newBad, sizeof(uint), cudaMemcpyHostToDevice));
//
//			kso::img::dspk::calc_norm_0<<<blocks, threads>>>(norm_d, gm_d, newBad_d, sz, ksz1);
//			kso::img::dspk::calc_norm_1<<<blocks, threads>>>(tmp_d, norm_d, sz, ksz1);
//			kso::img::dspk::calc_norm_2<<<blocks, threads>>>(norm_d, tmp_d, sz, ksz1);
//
//			kso::img::dspk::calc_gdev_0<<<blocks, threads>>>(gdev_d, dt_d, gm_d, sz, ksz1);
//			kso::img::dspk::calc_gdev_1<<<blocks, threads>>>(tmp_d, gdev_d, sz, ksz1);
//			kso::img::dspk::calc_gdev_2<<<blocks, threads>>>(gdev_d, tmp_d, dt_d, gm_d, norm_d, sz, ksz1);
//
//			kso::img::dspk::calc_nsd_0<<<blocks, threads>>>(nsd_d, gdev_d, sz, ksz1);
//			kso::img::dspk::calc_nsd_1<<<blocks, threads>>>(tmp_d, nsd_d, sz, ksz1);
//			kso::img::dspk::calc_nsd_2<<<blocks, threads>>>(nsd_d, tmp_d, norm_d, sz, ksz1);
//
//			kso::img::dspk::calc_gm<<<blocks, threads>>>(gm_d, gdev_d, nsd_d, dt_d, std_dev, newBad_d, sz, ksz1);
//
//			CHECK(cudaMemcpy(newBad, newBad_d, sizeof(uint), cudaMemcpyDeviceToHost));
//			cout << "Iteration " << iter << ": found " << *newBad << " bad pixels\n";
//			totBad = totBad + *newBad;
//
//			if(*newBad == 0){	// stop if we're not finding any pixels
//				break;
//			}
//
//		}


//
//		ksz = 3;
//
//		kso::img::dspk::calc_lmn_0<<<blocks, threads>>>(norm_d, gm_d, newBad_d, sz, ksz1);
//		kso::img::dspk::calc_lmn_1<<<blocks, threads>>>(tmp_d, norm_d, sz, ksz1);
//		kso::img::dspk::calc_lmn_2<<<blocks, threads>>>(norm_d, tmp_d, sz, ksz1);
//
//		float * gdt_d = gdev_d;	// reuse neighborhood mean memory
//		kso::img::dspk::calc_gdt<<<blocks, threads>>>(gdt_d, dt_d, gm_d, sz);
//
//		float * tp;	 // temporary pointer
//
//		Niter = 10;
//
//
//
//		for(uint iter = 0; iter < Niter; iter++){
//
//			// switch locations of temp and data buffer so this for loop works right
//			tp = gdt_d;
//			gdt_d = tmp_d;
//			tmp_d = tp;
//
//			kso::img::dspk::calc_gdt_0<<<blocks, threads>>>(gdt_d, tmp_d, gm_d, sz, ksz1);
//			kso::img::dspk::calc_gdt_1<<<blocks, threads>>>(tmp_d, gdt_d, sz, ksz1);
//			kso::img::dspk::calc_gdt_2<<<blocks, threads>>>(gdt_d, tmp_d, dt_d, gm_d, norm_d, sz, ksz1);
//
//			cout << "Iteration " << iter << endl;
//
//		}


		CHECK(cudaDeviceSynchronize());

		// copy back from devicecudaMemcpyDeviceToHost;
//		CHECK(cudaMemcpy(dt + b[s], gdt_d + b_d[s], m[s] * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(q1 + b[s], q1_d + b_d[s], m[s] * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(q2 + b[s], q2_d + b_d[s], m[s] * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(q3 + b[s], q3_d + b_d[s], m[s] * sizeof(float), cudaMemcpyDeviceToHost));


		cout << "Total bad pixels: " << totBad << endl;

	}



	return;


}

void denoise_ndarr(const np::ndarray & data, const np::ndarray & goodmap, const np::ndarray & hist, float tmin, float tmax, uint hsx, uint hsy, uint k_sz, uint Niter){

	// shape of input data
	dim3 sz;
	sz.z = data.get_shape()[0];
	sz.y = data.get_shape()[1];
	sz.x = data.get_shape()[2];

	dim3 st;
	st.z = data.get_strides()[0] / sizeof(float);
	st.y = data.get_strides()[1] / sizeof(float);
	st.x = data.get_strides()[2] / sizeof(float);
	dim3 hsz(hsx, hsy, 0);

	// extract float data from numpy array
	float * dt = (float *) data.get_data();
	float * gm = (float *) goodmap.get_data();

	uint n_threads = 1;

	buf * db = new buf(dt, gm, sz, k_sz, hsz, n_threads);

	denoise(db, tmin, tmax, Niter);

}

//np::ndarray denoise_fits_file(py::str path, float med_dev, float std_dev, uint k_sz, uint Niter){
//
//
//	string cpath = "/kso/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits";
////	string cpath = "/kso/iris_l2_20141129_000738_3860009154_raster_t000_r00000.fits";
//
//	uint n_threads = 1;
//	uint max_sz = pow(2,30);	// 1 GB
//
//	buf * db = new buf(cpath, max_sz, k_sz, n_threads);
//
//	denoise(db, med_dev, std_dev, Niter);
//
//	py::object own = py::object();
//	py::tuple shape = py::make_tuple(db->sz.z, db->sz.y, db->sz.x);
//	py::tuple stride = py::make_tuple(db->sb.z, db->sb.y, db->sb.x);
//	np::dtype dtype = np::dtype::get_builtin<float>();
//
//	return np::from_data(db->dt, dtype, shape, stride, own);
//
//}

np::ndarray denoise_fits_file_quartiles(const np::ndarray & q2, const np::ndarray & hist, uint hsx, uint hsy, uint k_sz){

	uint Niter = 1;

	string cpath = "/kso/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits";
//	string cpath = "/kso/iris_l2_20141129_000738_3860009154_raster_t000_r00000.fits";

	dim3 hsz(hsx, hsy, 0);

	uint n_threads = 1;
	uint max_sz = pow(2,30);	// 1 GB

	buf * db = new buf(cpath, max_sz, k_sz, hsz, n_threads);
	db->q2 = (float *)q2.get_data();
	db->ht = (float *)hist.get_data();

	float tmax = 99.9;
	float tmin = 0.01;


	denoise(db, tmin, tmax, Niter);

	py::object own = py::object();
	py::tuple shape = py::make_tuple(db->sz.z, db->sz.y, db->sz.x);
	py::tuple stride = py::make_tuple(db->sb.z, db->sb.y, db->sb.x);
	np::dtype dtype = np::dtype::get_builtin<float>();

	return np::from_data(db->dt, dtype, shape, stride, own);

}


}

}

}







