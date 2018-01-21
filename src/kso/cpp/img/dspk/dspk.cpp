
#include "dspk.h"

using namespace std;

void kso::img::dspk::remove_noise_3D(const np::ndarray & cube){

//	np::dtype np_double = np::dtype::get_builtin<double>();
//	cube.astype(np_double);

	float * data = (float *) cube.get_data();

	cout << data[0] << endl;

	cout << cube.get_shape()[0] << endl;

	printf("hello world\n");

}

BOOST_PYTHON_MODULE(dspk){

//	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

    boost::python::def("remove_noise_3D", kso::img::dspk::remove_noise_3D);

}


