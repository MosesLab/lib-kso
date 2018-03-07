
#include "pyboost.h"
#include "img/dspk/dspk.h"
#include "instrument/IRIS/read_fits.h"

BOOST_PYTHON_MODULE(libkso_cuda){

	Py_Initialize();
	np::initialize();   // only needed if you use numpy in the interface

	py::def("denoise_ndarr", kso::img::dspk::denoise_ndarr);
	py::def("read_fits_raster_ndarr", kso::instrument::IRIS::read_fits_raster_ndarr);
	py::def("read_fits_file", kso::img::dspk::read_fits_file);
	py::def("denoise_fits_file_quartiles", kso::img::dspk::denoise_fits_file_quartiles);


}
