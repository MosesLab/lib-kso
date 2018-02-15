
#include "test.h"

namespace kso {

namespace img {

namespace dspk {

namespace test {

void test_unitialized(){

	// shape of input data
	dim3 sz;
	sz.z = 256;
	sz.y = 256;
	sz.x = 256;
	uint sz3 = sz.x * sz.y * sz.z;

	dim3 st;
	st.x = 1;
	st.y = st.x * sz.x;
	st.z = st.y * sz.y;

	float * dt = new float[sz3];
	float * dn = new float[sz3];

	float std_dev = 3.0;
	float med_dev = 10.0;
	uint Niter = 5;
	uint k_sz = 5;

	uint n_threads = 10;

	buf * db = new buf(dt, dn, sz, k_sz, n_threads);

	denoise(db, med_dev, std_dev, Niter);

}

}

}

}

}
