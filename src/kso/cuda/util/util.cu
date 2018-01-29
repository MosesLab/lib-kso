
#include "util.h"

namespace kso {

namespace util {

void enum_device(){

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
	    cudaDeviceProp deviceProp;
	    cudaGetDeviceProperties(&deviceProp, device);
	    printf("Device %d has compute capability %d.%d.\n",
	           device, deviceProp.major, deviceProp.minor);
	    printf("Device %d has %.0f MiB of memory\n", device, deviceProp.totalGlobalMem / pow(2,20));
	}

}

}

}
