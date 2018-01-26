/*
 * math.h
 *
 *  Created on: Jan 25, 2018
 *      Author: byrdie
 */

#ifndef UTIL_MATH_H_
#define UTIL_MATH_H_

#include "../util/dim3.h"

namespace ku = kso::util;

namespace kso {

namespace math {

void add(float * out, float * in_0, float * in_1, ku::dim3 sz);
void sub(float * out, float * in_0, float * in_1, ku::dim3 sz);
void mul(float * out, float * in_0, float * in_1, ku::dim3 sz);
void div(float * out, float * in_0, float * in_1, ku::dim3 sz);


}

}


#endif /* UTIL_MATH_H_ */
