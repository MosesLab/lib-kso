
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

size_t get_device_mem(uint device) {

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	size_t tot_mem = deviceProp.totalGlobalMem;

	return tot_mem;
}

}

}
