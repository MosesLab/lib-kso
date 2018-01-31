
#include "stride.h"

using namespace std;

namespace kso {

namespace util {

stride::stride(uint _axis_sz, uint _buf_sz, uint _halo_sz){

	axis_sz = _axis_sz;
	buf_sz = _buf_sz;
	halo_sz = _halo_sz;

	if (buf_sz > axis_sz) buf_sz = axis_sz;
	uint eff_sz = (buf_sz - 2 * halo_sz);		// Throw exception if negative
	num_strides = ceil((float)(axis_sz - buf_sz) / (float) eff_sz) + 1;

	A.resize(num_strides, 0);
	a.resize(num_strides, 0);
	L.resize(num_strides, 0);
	l.resize(num_strides, 0);

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

}



}

}


