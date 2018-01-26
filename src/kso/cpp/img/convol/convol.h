/*
 * convol.h
 *
 *  Created on: Jan 24, 2018
 *      Author: byrdie
 */

#ifndef IMG_CONVOL_CONVOL_H_
#define IMG_CONVOL_CONVOL_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "util/dim3.h"

namespace ku = kso::util;

namespace kso {

namespace img {

namespace convol {


void sconv_x(float * krn, float * in, float * out, ku::dim3 sz, uint k_sz);
void sconv_y(float * krn, float * in, float * out, ku::dim3 sz, uint k_sz);
void sconv_z(float * krn, float * in, float * out, ku::dim3 sz, uint k_sz);

}

}

}



#endif /* IMG_CONVOL_CONVOL_H_ */
