
#include "read_fits.h"

namespace kso {

namespace instrument {

namespace IRIS {

using namespace CCfits;

void read_fits_raster(std::string path, float * buf){

	std::auto_ptr<FITS> pInfile(new FITS(path,Read,true));

	PHDU& image = pInfile->pHDU();

	std::valarray<unsigned long>  contents;

	// read all user-specifed, coordinate, and checksum keys in the image
	image.readAllKeys();

	image.read(contents);

	// this doesn't print the data, just header info.
	std::cout << image << std::endl;

}



}

}

}
