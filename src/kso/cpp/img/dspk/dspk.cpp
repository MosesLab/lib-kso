
#include "dspk.h"

using namespace std;

void kso::img::dspk::remove_noise_3D(const np::ndarray & cube, float std_dev, uint k_sz, uint Niter){


	// extract float data from numpy array
	float * dt = (float *) cube.get_data();

	// shape of input data
	uint sz_t = cube.get_shape()[0];
	uint sz_y = cube.get_shape()[1];
	uint sz_l = cube.get_shape()[2];

	// stride of input data
	uint n_t = cube.get_strides()[0] / sizeof(float);
	uint n_y = cube.get_strides()[1] / sizeof(float);
	uint n_l = cube.get_strides()[2] / sizeof(float);
	uint n = n_t * n_y * n_l;

	cout << n_t << "\n";
	cout << n_y << "\n";
	cout << n_l << "\n";

	// initialize goodmap
	float * gm = new float[n];
	fill(gm, gm + n, 1.0);

	// initialize neighborhood mean kernel
	float K_sz = k_sz * k_sz * k_sz;	// total size of the kernel
	float * nm_krn = new float[K_sz];
	fill(nm_krn, nm_krn + K_sz, 1.0);

	// calculate offset for kernel
	uint ks2 = k_sz / 2;


	// Number of identification iterations
	for(uint iter = 0; iter < Niter; iter++){

		// loop over time
		for(uint t = ks2; t < sz_t - ks2; t++){

			// loop over space
			for(uint y = ks2; y < sz_y - ks2; y++){

				// loop pver spectrum
				for(uint l = ks2; l < sz_l - ks2; l++){

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

								uint X = t - ks2 + i;
								uint Y = y - ks2 + j;
								uint Z = l - ks2 + k;

								double gm_0 = gm[n_t * X + n_y * Y + n_l * Z];
								double dt_0 = dt[n_t * X + n_y * Y + n_l * Z];

								nm_norm = nm_norm + gm_0;
								nm = nm + (gm_0 * dt_0);

							}

						}

					}

					// renormalize neighborhood mean
					nm = nm / nm_norm;

					// subtract neighborhood mean from data
					float dt_tyl = dt[n_t * t + n_y * y + n_l * l];

					// calculate deviation
					float dev = dt_tyl - nm;

					// initialize neighborhood standard deviation
					float ns = 0.0;
					float ns_norm = 0.0;

					// Neighborhood standard deviation convolution
					// convolve over time
					for(uint i = 0; i < k_sz; i++){

						// convolve over space
						for(uint j = 0; j < k_sz; j++){

							// convolve over spectrum
							for(uint k = 0; k < k_sz; k++){



							}

						}

					}



				}

			}

		}
	}


}

BOOST_PYTHON_MODULE(dspk){

	//	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	boost::python::def("remove_noise_3D", kso::img::dspk::remove_noise_3D);

}


