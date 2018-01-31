/*
 * stride.h
 *
 *  Created on: Jan 29, 2018
 *      Author: byrdie
 */

#ifndef STRIDE_H_
#define STRIDE_H_

#include <vector>

namespace kso {

namespace util {

class stride {
public:
	uint axis_sz;		// original size of striding axis
	uint buf_sz;		// number of elements in target buffer
	uint halo_sz;		// extent of the halo in elements (half-size of convolution kernel)

	uint axis_st;		// original stride of the input axis

	uint num_strides;		// number of strides

	uint * A;		// host write indices
	uint * a;		// host read indices
	uint * L;		// host write lengths
	uint * l;		// host read lengths
	uint * a_d;	// device write indices

	uint * B;
	uint * b;
	uint * M;
	uint * m;
	uint * b_d;

	stride(uint _axis_sz, uint _buf_sz, uint _halo_sz, uint _axis_stride);


};


}

}


#endif /* STRIDE_H_ */
