/*
 * read_fits.h
 *
 *  Created on: Jan 30, 2018
 *      Author: byrdie
 */

#ifndef READ_FITS_H_
#define READ_FITS_H_

#include <iostream>
#include <string>

//#include <CCfits/CCfits>
#include "fitsio.h"


#include "pyboost.h"

namespace kso {

namespace instrument {

namespace IRIS {

dim3 read_fits_raster(std::string path, float * buf);
void read_fits_raster_ndarr(np::ndarray & nd_buf, np::ndarray & nd_sz);
//void read_fits_raster(py::str, np::ndarray & buf);

}

}

}





#endif /* READ_FITS_H_ */
