
#include "test.h"

using namespace std;

namespace kso {

namespace instrument {

namespace IRIS {

namespace test {

void const_path(){

	string path = "/kso/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits";

	float * buf = new float[256];

	kso::instrument::IRIS::read_fits_raster(path, buf);

}

}

}

}

}
