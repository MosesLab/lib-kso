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

	uint num_strides;		// number of strides

	std::vector<uint> A;		// host write indices
	std::vector<uint> a;		// host read indices
	std::vector<uint> L;		// host write lengths
	std::vector<uint> l;		// host read lengths
	std::vector<uint> a_d;	// device write indices

	stride(uint _axis_sz, uint _buf_sz, uint _halo_sz);


};


}

}


#endif /* STRIDE_H_ */
