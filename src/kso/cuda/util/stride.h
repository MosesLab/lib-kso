/*
 * stride.h
 *
 *  Created on: Jan 29, 2018
 *      Author: byrdie
 */

#ifndef STRIDE_H_
#define STRIDE_H_


namespace kso {

namespace util {

namespace stride {

uint get_num_strides(uint axis_sz, uint buf_sz, uint halo_sz);
void get_strides(uint axis_sz, uint buf_sz, uint halo_sz, uint * in_ind, uint * out_ind, uint * in_len, uint * out_len, uint * dev_out_ind);

}

}

}


#endif /* STRIDE_H_ */
