
#include "dspk.h"

using namespace std;

void kso::img::dspk::remove_noise_3D(){

//	cout << cube.get_shape() << endl;

	printf("hello world\n");

}

BOOST_PYTHON_MODULE(dspk){

    boost::python::def("remove_noise_3D", kso::img::dspk::remove_noise_3D);

}


