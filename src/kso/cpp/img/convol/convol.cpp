/*
 * convol.cpp
 *
 *  Created on: Jan 24, 2018
 *      Author: byrdie
 */

#include "convol.h"

using namespace std;

namespace kso {

namespace img {

namespace convol {


void sconv_x(float * krn, float * in, float * out, uint k_sz, ku::dim3 sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;


	// retrieve sizes
	uint sz_x = sz.x;
	uint sz_y = sz.y;
	uint sz_z = sz.z;

	// compute stride sizes
	uint n_y = sz_x;
	uint n_z = n_y * sz_y;

	for(uint z = 0; z < sz_z; z++){
		for(uint y = 0; y < sz_y; y++){
			for(uint x = 0; x < sz_x; x++){

				float sum = 0.0;

				for(uint i = 0; i < k_sz; i++){


					// calculate offset
					uint I = x - ks2 + i;

					// truncate kernel if we're over the edge
					if(I > (sz_x - 1)){
						continue;
					}

					// load from memory
					float krn_0 = krn[n_z * z + n_y * y + I];
					float in_0 = in[n_z * z + n_y * y + I];

					// update sum
					sum = sum + (in_0 * krn_0);


				}

				// store result
				out[n_z * z + n_y * y + x] = sum;

			}
		}
	}



}

void sconv_y(float * krn, float * in, float * out, uint k_sz, ku::dim3 sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;


	// retrieve sizes
	uint sz_x = sz.x;
	uint sz_y = sz.y;
	uint sz_z = sz.z;

	// compute stride sizes
	uint n_y = sz_x;
	uint n_z = n_y * sz_y;

	for(uint z = 0; z < sz_z; z++){
		for(uint y = 0; y < sz_y; y++){
			for(uint x = 0; x < sz_x; x++){

				float sum = 0.0;

				for(uint j = 0; j < k_sz; j++){


					// calculate offset
					uint J = x - ks2 + j;

					// truncate kernel if we're over the edge
					if(J > (sz_y - 1)){
						continue;
					}

					// load from memory
					float krn_0 = krn[n_z * z + n_y * J + x];
					float in_0 = in[n_z * z + n_y * J + x];

					// update sum
					sum = sum + (in_0 * krn_0);


				}

				// store result
				out[n_z * z + n_y * y + x] = sum;

			}
		}
	}



}


void sconv_z(float * krn, float * in, float * out, uint k_sz, ku::dim3 sz){

	// calculate offset for kernel
	uint ks2 = k_sz / 2;


	// retrieve sizes
	uint sz_x = sz.x;
	uint sz_y = sz.y;
	uint sz_z = sz.z;

	// compute stride sizes
	uint n_y = sz_x;
	uint n_z = n_y * sz_y;

	for(uint z = 0; z < sz_z; z++){
		for(uint y = 0; y < sz_y; y++){
			for(uint x = 0; x < sz_x; x++){

				float sum = 0.0;

				for(uint k = 0; k < k_sz; k++){


					// calculate offset
					uint K = x - ks2 + k;

					// truncate kernel if we're over the edge
					if(K > (sz_z - 1)){
						continue;
					}

					// load from memory
					float krn_0 = krn[n_z * K + n_y * y + x];
					float in_0 = in[n_z * K + n_y * y + x];

					// update sum
					sum = sum + (in_0 * krn_0);


				}

				// store result
				out[n_z * z + n_y * y + x] = sum;

			}
		}
	}



}

}

}

}
