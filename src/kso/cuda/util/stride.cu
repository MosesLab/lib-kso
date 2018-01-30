
#include "stride.h"

namespace kso {

namespace util {

namespace stride {

uint get_num_strides(uint axis_sz, uint buf_sz, uint halo_sz){

	if (buf_sz > axis_sz) buf_sz = axis_sz;
	uint eff_sz = (buf_sz - 2 * halo_sz);		// Throw exception if negative
	return ceil((float)(axis_sz - buf_sz) / (float) eff_sz) + 1;

}
void get_strides(uint axis_sz, uint buf_sz, uint halo_sz, uint * in_ind, uint * out_ind, uint * in_len, uint * out_len, uint * dev_out_ind){

	uint * A = in_ind;
	uint * a = out_ind;
	uint * L = in_len;
	uint * l = out_len;
	uint * a_d = dev_out_ind;

	if (buf_sz > axis_sz) buf_sz = axis_sz;
	uint eff_sz = (buf_sz - 2 * halo_sz);		// Throw exception if negative
	uint s_sz = get_num_strides(axis_sz, buf_sz, halo_sz);



	uint i;
	for(i = 0; i < s_sz; i++){



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


		if (i == (s_sz - 1)) {
			L[i] = axis_sz - A[i];
			l[i] = axis_sz - a[i];
		}


	}


}


}

}

}
