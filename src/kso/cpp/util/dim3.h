/*
 * dim3.h
 *
 *  Created on: Jan 25, 2018
 *      Author: byrdie
 */

#ifndef IMG_UTIL_DIM3_H_
#define IMG_UTIL_DIM3_H_

#include <stdio.h>
#include <stdlib.h>

namespace kso {

namespace util {

class dim3{
public:
	uint x;
	uint y;
	uint z;
	dim3(uint _x, uint _y, uint _z);
};

}

}




#endif /* IMG_UTIL_DIM3_H_ */
