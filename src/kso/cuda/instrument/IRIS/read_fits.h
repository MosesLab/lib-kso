/*
 * read_fits.h
 *
 *  Created on: Jan 30, 2018
 *      Author: byrdie
 */

#ifndef READ_FITS_H_
#define READ_FITS_H_

#include <string>

#include <CCfits/CCfits>




namespace kso {

namespace instrument {

namespace IRIS {

void read_fits_raster(std::string path, float * buf);
//void read_fits_raster(py::str, np::ndarray & buf);

}

}

}





#endif /* READ_FITS_H_ */
