
#include "stride.h"

using namespace std;

namespace kso {

namespace util {

stride::stride(uint _axis_sz, uint _buf_sz, uint _halo_sz, uint _axis_stride){

	axis_sz = _axis_sz;
	buf_sz = _buf_sz;
	halo_sz = _halo_sz;

	axis_st = _axis_stride;

	if (buf_sz > axis_sz) buf_sz = axis_sz;
	uint eff_sz = (buf_sz - 2 * halo_sz);		// Throw exception if negative
	num_strides = ceil((float)(axis_sz - buf_sz) / (float) eff_sz) + 1;

	A = new uint[num_strides];
	a = new uint[num_strides];
	L = new uint[num_strides];
	l = new uint[num_strides];
	B = new uint[num_strides];
	b = new uint[num_strides];
	M = new uint[num_strides];
	m = new uint[num_strides];
	a_d = new uint[num_strides];
	b_d = new uint[num_strides];

	uint i;
	for(i = 0; i < num_strides; i++){


		if(i == 0){
			A[0] = 0;
			a[0] = 0;
			L[0] = buf_sz;
			l[0] = buf_sz - halo_sz;
			a_d[0] = 0;
		} else {
			A[i] = A[i - 1] + eff_sz;
			a[i] = a[i - 1] + l[i - 1];
			a_d[i] = halo_sz;
			L[i] = buf_sz;
			l[i] = eff_sz;
		}


		if (i == (num_strides - 1)) {
			L[i] = axis_sz - A[i];
			l[i] = axis_sz - a[i];
		}


	}


	// calculate the parameters using the actual stride
	for(i = 0; i < num_strides; i++){

		B[i] = A[i] * axis_st;
		b[i] = a[i] * axis_st;
		b_d[i] = a_d[i] * axis_st;
		M[i] = L[i] * axis_st;
		m[i] = l[i] * axis_st;

	}

}



}

}


