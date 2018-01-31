
#include "pyboost.h"
#include "img/dspk/dspk.h"
#include "instrument/IRIS/read_fits.h"

BOOST_PYTHON_MODULE(libkso_cuda){

	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	py::def("locate_noise_3D", kso::img::dspk::locate_noise_3D);


}
