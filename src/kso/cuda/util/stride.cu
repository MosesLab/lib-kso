
#include "stride.h"

namespace kso {

namespace util {

namespace stride {

uint get_num_strides(uint axis_sz, uint buf_sz, uint halo_sz){

	return ceil((double)(axis_sz - halo_sz) / (double)(buf_sz - 2 * halo_sz));

}
void get_strides(uint axis_sz, uint buf_sz, uint halo_sz, uint * in_ind, uint * out_ind, uint * in_len, uint * out_len, uint * dev_out_ind){

	uint * A = in_ind;
	uint * a = out_ind;
	uint * L = in_len;
	uint * l = out_len;
	uint * a_d = dev_out_ind;

	uint eff_sz = (buf_sz - 2 * halo_sz);		// Throw exception if negative
	uint s_sz = get_num_strides(axis_sz, buf_sz, halo_sz);

	A[0] = 0;
	a[0] = 0;
	L[0] = buf_sz;
	l[0] = buf_sz - halo_sz;
	a_d[0] = 0;

	uint i;
	for(i = 1; i < (s_sz - 1); i++){

		A[i] = A[i - 1] + eff_sz;
		a[i] = a[i - 1] + l[i - 1];
		L[i] = buf_sz;
		l[i] = eff_sz;
		a_d[i] = halo_sz;


	}

	A[i] = A[i - 1] + eff_sz;
	a[i] = a[i - 1] + l[i - 1];
	L[i] = axis_sz - A[i];
	l[i] = axis_sz - a[i];
	a_d[i] = halo_sz;

}


}

}

}
