/*
 * dspk_cuda.h
 *
 *  Created on: Jan 19, 2018
 *      Author: byrdie
 */

#ifndef DSPK_CUDA_H_
#define DSPK_CUDA_H_


#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#include <boost/python.hpp>

int dspk(void);

BOOST_PYTHON_MODULE(dspk_cuda)
{
    using namespace boost::python;

    def("dspk", dspk);
}



#endif /* DSPK_CUDA_H_ */
