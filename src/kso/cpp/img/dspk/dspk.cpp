

#include <stdio.h>
#include <stdlib.h>

#include "dspk.h"


void kso::img::dspk::remove_noise(){

	printf("hello world\n");

}

BOOST_PYTHON_MODULE(dspk){

    boost::python::def("remove_noise", kso::img::dspk::remove_noise);

}
