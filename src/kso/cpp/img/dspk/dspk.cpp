
#include "dspk.h"

using namespace std;

np::ndarray kso::img::dspk::remove_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter){


	cout << std_dev << endl;

	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// shape of input data
	uint sz_t = cube.get_shape()[0];
	uint sz_y = cube.get_shape()[1];
	uint sz_l = cube.get_shape()[2];
	uint sz = sz_t * sz_y * sz_l;

	// stride of input data
	uint n_t = cube.get_strides()[0] / sizeof(float);
	uint n_y = cube.get_strides()[1] / sizeof(float);
	uint n_l = cube.get_strides()[2] / sizeof(float);
	uint n = n_t * n_y * n_l;

	cout << n_t << "\n";
	cout << n_y << "\n";
	cout << n_l << "\n";

	// initialize goodmap
	float * gm = new float[sz];
	fill(gm, gm + n, 1.0);

	// initialize neighborhood mean kernel
	uint K_sz = k_sz * k_sz * k_sz;	// total size of the kernel
	float * nm_krn = new float[K_sz];
	fill(nm_krn, nm_krn + K_sz, 1.0);

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// limits on main (outer) 3D loop. Need to account for kernel size
	uint T0 = 2 * ks2;
	uint T9 = sz_t - 2 * ks2;
	uint Y0 = T0;
	uint Y9 = T9;
	uint L0 = T0;
	uint L9 = T9;

	// storage for the number of bad pixels found on each iteration
	uint new_bad = 0;

	// Number of identification iterations
	for(uint iter = 0; iter < Niter; iter++){

		new_bad = 0;	// reset the number of bad pixels found for this iteration

		// loop over time
		for(uint t = T0; t < T9; t++){

			// loop over space
			for(uint y = Y0; y < Y9; y++){

				// loop over spectrum
				for(uint l = L0; l < L9; l++){

					// initialize neighborhood standard deviation
					float nsd = 0.0;
					float nsd_norm = 0.0;

					// storage for deviation on this current pixel
					float this_dev;

					// Neighborhood standard deviation convolution
					// convolve over time
					for(uint a = 0; a < k_sz; a++){

						// convolve over space
						for(uint b = 0; b < k_sz; b++){

							// convolve over spectrum
							for(uint c = 0; c < k_sz; c++){

								// calculate offsets
								uint A = t - ks2 + a;
								uint B = y - ks2 + b;
								uint C = l - ks2 + c;

								// initialize neighborhood mean
								float nm = 0.0;
								float nm_norm = 0.0;

								// Neighborhood mean convolution
								// convolve over time
								for(uint i = 0; i < k_sz; i++){

									// convolve over space
									for(uint j = 0; j < k_sz; j++){

										// convolve over spectrum
										for(uint k = 0; k < k_sz; k++){

											// calculate offsets
											uint I = A - ks2 + i;
											uint J = B - ks2 + j;
											uint K = C - ks2 + k;

											double gm_0 = gm[n_t * I + n_y * J + n_l * K];
											double dt_0 = dt[n_t * I + n_y * J + n_l * K];

											nm_norm = nm_norm + gm_0;
											nm = nm + (gm_0 * dt_0);

										}

									}

								}

								// renormalize neighborhood mean
								nm = nm / nm_norm;

								// load data from memory
								float dt_tyl = dt[n_t * A + n_y * B + n_l * C];

								// calculate deviation
								float dev = dt_tyl - nm;

								// calculate variance
								float var =  dev * dev;

								// load current coordinate from goodmap
								double gm_0 = gm[n_t * A + n_y * B + n_l * C];

								// update neighborhood standard deviation
								nsd_norm = nsd_norm + gm_0;
								nsd = nsd + (gm_0 * var);

								// save deviation for the center pixel
								if(A == t and B == y and C == l){
									this_dev = dev * gm_0;
								}


							}

						}

					}



					// renormalize neighborhood standard deviation
					nsd = sqrt(nsd / nsd_norm);

					// Check if a pixel is bad
					float good = gm[n_t * t + n_y * y + n_l * l];
					if(this_dev > (std_dev * nsd)){
						good = 0.0;
						new_bad = new_bad + 1;	// keep track of the number of bad pixels found each iteration
					}

					// update map of good pixels
					gm[n_t * t + n_y * y + n_l * l] = good;

				}

			}

		}

		cout << "Iteration " << iter << ": found " << new_bad << " bad pixels\n";

	}

	//	p::object own;
	//	np::ndarray gm_nd = np::from_data(gm, cube.get_dtype(), cube.get_shape(), cube.get_strides(), own);


	p::object gm_own = p::object();
	p::tuple gm_stride = p::make_tuple(n_t * sizeof(float), n_y * sizeof(float), n_l * sizeof(float));
	p::tuple gm_shape = p::make_tuple(sz_t, sz_y, sz_l);
	np::dtype gm_type = np::dtype::get_builtin<float>();
	np::ndarray gm_arr = np::from_data(gm, gm_type, gm_shape, gm_stride, gm_own);

	//	np::ndarray gm_nd = np::zeros(gm_stride, cube.get_dtype());

	//	copy(gm, gm + sz, reinterpret_cast<float *>(cube.get_data()));

	return gm_arr;


}

np::ndarray kso::img::dspk::locate_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter){


	cout << std_dev << endl;

	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// shape of input data
	uint sz_t = cube.get_shape()[0];
	uint sz_y = cube.get_shape()[1];
	uint sz_l = cube.get_shape()[2];
	uint sz = sz_t * sz_y * sz_l;

	// stride of input data
	uint n_t = cube.get_strides()[0] / sizeof(float);
	uint n_y = cube.get_strides()[1] / sizeof(float);
	uint n_l = cube.get_strides()[2] / sizeof(float);
	uint n = n_t * n_y * n_l;

	cout << n_t << "\n";
	cout << n_y << "\n";
	cout << n_l << "\n";

	// initialize goodmap
	float * gm = new float[sz];
	fill(gm, gm + n, 1.0);

	// initialize neighborhood mean
//	float * nm = new float[sz];

	// initialize neighborhood deviation
	float * dev = new float[sz];

	// initialize constant kernel
	uint K_sz = k_sz * k_sz * k_sz;	// total size of the kernel
	float * krn = new float[K_sz];
	fill(krn, krn + K_sz, 1.0);

	// calculate offset for kernel
	uint ks2 = k_sz / 2;

	// limits on main (outer) 3D loop. Need to account for kernel size
	uint T0 = ks2;
	uint T9 = sz_t - ks2;
	uint Y0 = T0;
	uint Y9 = T9;
	uint L0 = T0;
	uint L9 = T9;

	// storage for the number of bad pixels found on each iteration
	uint new_bad = 0;

	// Number of identification iterations
	for(uint iter = 0; iter < Niter; iter++){

		new_bad = 0;	// reset the number of bad pixels found for this iteration

		// -----------------------------------------------------------------------
		// NEIGHBORHOOD MEAN CONVOLUTION
		// loop over time
		for(uint t = 0; t < sz_t; t++){

			// loop over space
			for(uint y = 0; y < sz_y; y++){

				// loop over spectrum
				for(uint l = 0; l < sz_l; l++){

					// initialize neighborhood mean
					float mean = 0.0;
					float norm = 0.0;


					// convolve over time
					for(uint a = 0; a < k_sz; a++){

						// calculate offsets
						uint A = t - ks2 + a;

						// truncate the kernel if we're over the edge
						if(A < T0 or A > T9){
							continue;
						}

						// convolve over space
						for(uint b = 0; b < k_sz; b++){

							// calculate offset
							uint B = y - ks2 + b;

							// truncate kernel if we're over the edge
							if(B < Y0 or B > Y9) {
								continue;
							}

							// convolve over spectrum
							for(uint c = 0; c < k_sz; c++){

								// calculate offset
								uint C = l - ks2 + c;

								// truncate kernel if we're over the edge
								if(C < Y0 or C > Y9){
									continue;
								}

								// load from memory
								double gm_0 = gm[n_t * A + n_y * B + n_l * C];
								double dt_0 = dt[n_t * A + n_y * B + n_l * C];

								// update value of mean
								norm = norm + gm_0;
								mean = mean + (gm_0 * dt_0);

							}

						}

					}

					dev[n_t * t + n_y * y + n_l * l] =  dt[n_t * t + n_y * y + n_l * l] - (mean / norm);


				}

			}

		}

		// -----------------------------------------------------------------------
		// NEIGHBORHOOD STANDARD DEVIATION CONVOLUTION
		// loop over time
		for(uint t = 0; t < sz_t; t++){

			// loop over space
			for(uint y = 0; y < sz_y; y++){

				// loop over spectrum
				for(uint l = 0; l < sz_l; l++){

					// initialize neighborhood mean
					float std = 0.0;
					float norm = 0.0;


					// convolve over time
					for(uint a = 0; a < k_sz; a++){

						// calculate offsets
						uint A = t - ks2 + a;

						// truncate the kernel if we're over the edge
						if(A < T0 or A > T9){
							continue;
						}

						// convolve over space
						for(uint b = 0; b < k_sz; b++){

							// calculate offset
							uint B = y - ks2 + b;

							// truncate kernel if we're over the edge
							if(B < Y0 or B > Y9) {
								continue;
							}

							// convolve over spectrum
							for(uint c = 0; c < k_sz; c++){

								// calculate offset
								uint C = l - ks2 + c;

								// truncate kernel if we're over the edge
								if(C < Y0 or C > Y9){
									continue;
								}

								// load from memory
								double gm_0 = gm[n_t * A + n_y * B + n_l * C];
								double dt_0 = dt[n_t * A + n_y * B + n_l * C];
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

					// check if bad pixel
					if((dev_0 * gm_0) > (std_dev * std)){
						gm[n_t * t + n_y * y + n_l * l] = 0.0;	// update good pixel map
						new_bad++;	// update bad pixel count for this iteration
					}

				}

			}

		}


		cout << "Iteration " << iter << ": found " << new_bad << " bad pixels\n";

	}

	//	p::object own;
	//	np::ndarray gm_nd = np::from_data(gm, cube.get_dtype(), cube.get_shape(), cube.get_strides(), own);


	p::object gm_own = p::object();
	p::tuple gm_stride = p::make_tuple(n_t * sizeof(float), n_y * sizeof(float), n_l * sizeof(float));
	p::tuple gm_shape = p::make_tuple(sz_t, sz_y, sz_l);
	np::dtype gm_type = np::dtype::get_builtin<float>();
	np::ndarray gm_arr = np::from_data(gm, gm_type, gm_shape, gm_stride, gm_own);

	//	np::ndarray gm_nd = np::zeros(gm_stride, cube.get_dtype());

	//	copy(gm, gm + sz, reinterpret_cast<float *>(cube.get_data()));

	return gm_arr;


}

BOOST_PYTHON_MODULE(dspk){

	//	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	boost::python::def("remove_noise_3D", kso::img::dspk::remove_noise_3D);
	boost::python::def("locate_noise_3D", kso::img::dspk::locate_noise_3D);

}


